"""Do in-game NPCs answer from what they were given, or from generic priors?

    .venv/Scripts/python.exe evaluation/run_grounding.py [variant ...]

Every question here has its answer in the NPC's prompt: the campus facts and
personal facts from SampleGame/.../Systems/NPC/campus_facts.gd (parsed from
that file, so this can never drift from the game), or the NPC's background
line from npc_director.gd. A reply is *grounded* when it contains the answer
("first floor" for the canteen), *generic* when it answers something else
("There's a café on the corner"), and a *refusal* when it declines.

Requests are built exactly as the game builds them: persona, event, other
guests, facts, fact demos, max_tokens 64, no transcript (first question).

Two phrasings of every question, written together before any run:
  dev   used to design any fix
  test  reported; not used to choose anything (same author, so not blind)

Default run: `block` against `server`, both through the real /chat handler.

  block      the layout before fact retrieval: every fact in the system prompt,
             no normalisation, no repairs (LEGACY below)
  server     the dialogue pipeline as the game gets it (TurnConfig())
  sys_retry  only the retrieved facts, in the system prompt, plus a refusal
             retry -- about as accurate as `server`, but it changes the start
             of the cached prompt on every turn (see `latency`)

`run_grounding.py latency` times an eight-turn conversation under block,
sys_block (retrieved facts in the system prompt) and server.

The exploratory placements tried on the way to `server` (facts inside the
player's message, both early and late, and so on) are recorded in
paper/NUMBERS.md; they were measured on the pre-refactor server and are not
kept as code.
"""
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import gguf_server as gs  # noqa: E402
from dialogue import TurnConfig, composer, guard, knowledge  # noqa: E402
from dialogue import intent as intents  # noqa: E402
from dialogue.text import normalize  # noqa: E402
from run_player_memory import EVENT, EVENT_LINE, NPCS  # noqa: E402

FACTS_GD = REPO_ROOT / "SampleGame" / "new-game-project" / "Systems" / "NPC" / "campus_facts.gd"
OUT_PATH = REPO_ROOT / "evaluation" / "results" / "grounding_results.json"
ROLES = {"Prof. Adeyemi": "faculty", "Ms. Okafor": "student counsellor", "Nadia": "campus store",
         "Halvorsen": "head of department", "Officer Reyes": "campus liaison"}
REPLY_TOKENS = 64  # npc_director.gd REPLY_TOKENS


def _strings(block: str) -> list:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', block)


def load_campus_facts():
    src = FACTS_GD.read_text(encoding="utf-8")
    shared = _strings(re.search(r"const SHARED := \[(.*?)\n\]", src, re.S).group(1))
    personal_src = re.search(r"const PERSONAL := \{(.*?)\n\}", src, re.S).group(1)
    personal = {m.group(1): _strings(m.group(2))
                for m in re.finditer(r'"([^"]+)": \[(.*?)\n\t\]', personal_src, re.S)}
    pay_src = re.search(r"const PAY_ANSWER := \{(.*?)\n\}", src, re.S).group(1)
    pay = dict(re.findall(r'"([^"]+)": "((?:[^"\\]|\\.)*)"', pay_src))
    demo_q, demo_a = re.search(
        r'"content": "(how many students[^"]*)"\},\s*\{"role": "assistant",\s*"content": "([^"]*)"',
        src, re.S).groups()
    return shared, personal, pay, (demo_q, demo_a)


SHARED, PERSONAL, PAY, STUDENT_DEMO = load_campus_facts()


def fact_demos(npc):
    """Same as CampusFacts.demos_for()."""
    demos = [{"role": "user", "content": STUDENT_DEMO[0]},
             {"role": "assistant", "content": STUDENT_DEMO[1]}]
    if npc in PAY:
        for q in ("what do you earn", "What is your salary?"):
            demos += [{"role": "user", "content": q}, {"role": "assistant", "content": PAY[npc]}]
    return demos


# (dev phrasing, test phrasing, answer regex). Asked of every NPC.
SHARED_Q = [
    ("where's the canteen?", "where can I grab lunch?", r"first floor"),
    ("where are the computer labs?", "which floor are the teaching labs on?", r"\bfirst\b"),
    ("where are the staff offices?", "where would I find a lecturer's office?", r"\bsecond\b"),
    ("when does term start?", "when do classes begin in the autumn?", r"september"),
    ("how many lecturers work here?", "how big is the teaching staff?", r"\b34\b|thirty-four"),
    ("what time does the open day end?", "how long is the open day on for?",
     r"\b4 ?(pm|o'clock)\b|\bfour\b|\b10 ?(am)? (to|until|-) ?4|six hours"),
    ("where are the lecture rooms?", "where's the seminar hall?", r"ground"),
]
# Per NPC, from PERSONAL facts or the background line in npc_director.gd.
PERSONAL_Q = {
    "Prof. Adeyemi": [
        ("which modules do you teach?", "what courses do you run?",
         r"algorithm|compiler|cs ?2011|cs ?3040"),
        ("how many projects do you supervise?", "how many final-year students do you look after?",
         r"\bsix\b|\b6\b"),
        ("how did you end up teaching here?", "where did you work before this?",
         r"industry|doctorate|phd"),
    ],
    "Halvorsen": [
        ("how long have you been head of department?", "how many years have you run the department?",
         r"\bthree\b|\b3\b"),
        ("do you still teach?", "which module do you teach?", r"cs ?1002|programming"),
        ("how many people report to you?", "how big is your team?", r"\b34\b|\b45\b|\b11\b"),
    ],
    "Officer Reyes": [
        ("how long have you been the campus liaison?", "how many years have you been doing this job?",
         r"\bsix\b|\b6\b"),
        ("what are your working hours?", "when does your shift start?", r"\b7 ?(am|a\.m\.)?\b|seven"),
        ("what do you deal with most?", "what's the most common problem on campus?",
         r"lost property|bike"),
    ],
    "Ms. Okafor": [
        ("how long have you worked in student welfare?", "how many years have you been a counsellor?",
         r"\bnine\b|\b9\b"),
        ("when are the drop-in sessions?", "can I just turn up without an appointment?",
         r"tuesday|thursday|drop-in|no appointment"),
        ("how many students do you support?", "how big is your caseload?", r"\b40\b|forty"),
    ],
    "Nadia": [
        ("how did you get the shop?", "who ran the shop before you?", r"\baunt\b"),
        ("when does the shop open?", "what are your opening hours?", r"\b7 ?(am|a\.m\.)?\b|seven"),
        ("who helps you run the shop?", "do you have any staff?", r"\btwo\b|\b2\b|student|part-time"),
    ],
}
REFUSAL = re.compile(r"\b(don't know|do not know|not sure|no idea|not something|can't tell|cannot tell"
                     r"|don't have|not allowed|i'm not able)\b", re.IGNORECASE)


def build_request(npc, message):
    arch, occupation, intro, job_line, background = NPCS[npc]
    others = ", ".join("%s (%s)" % (n, NPCS[n][1]) for n in NPCS if n != npc)  # occupation, as npc_director.gd sends
    return gs.ChatRequest(
        archetype=arch, message=message, max_tokens=REPLY_TOKENS, name=npc,
        occupation=occupation, intro=intro, job_line=job_line, background=background,
        situation=EVENT, event_line=EVENT_LINE, others=others,
        facts="\n".join(SHARED + PERSONAL.get(npc, [])),
        fact_demos=[gs.ChatTurn(**t) for t in fact_demos(npc)])


def questions(split):
    idx = 0 if split == "dev" else 1
    for npc in NPCS:
        for q in SHARED_Q + PERSONAL_Q[npc]:
            yield npc, q[idx], q[2], ("shared" if q in SHARED_Q else "personal")


# The server before fact retrieval and the reply guard existed.
LEGACY = TurnConfig(normalize=False, fact_retrieval=False, memory_turn="always", repairs=False,
                    deloop=False, speech_shape=False)
CONFIGS = {"block": LEGACY, "server": TurnConfig()}


def chat_with(config, req):
    """The real /chat handler, run under `config` instead of the game's."""
    saved = gs.TURN_CONFIG
    gs.TURN_CONFIG = config
    try:
        return gs.chat(req)
    finally:
        gs.TURN_CONFIG = saved


def retrieved(npc, req):
    text = normalize(req.message)
    return knowledge.select(text, knowledge.fact_pool(req.facts.splitlines(), req.job_line),
                            knowledge.split_sentences(req.background))


def answer(npc, message, variant):
    req = build_request(npc, message)
    if variant in CONFIGS:
        out = chat_with(CONFIGS[variant], req)
        LEAKS[variant] = LEAKS.get(variant, 0) + bool(out.leaked_fact_ids)
        return out.response
    if variant == "sys_retry":
        chosen = retrieved(npc, req)
        if chosen:
            req.facts = "\n".join(chosen)
        inp = gs.to_turn_input(req)
        text = normalize(message)
        msgs = composer.compose(inp, text, intents.classify(text), {}, [], LEGACY)
        generate = gs.LlamaGenerator(gs.pool.get(req.archetype)[0])
        reply = guard.clean_basic(generate(msgs, REPLY_TOKENS))
        if chosen and REFUSAL.search(reply):
            reply = guard.clean_basic(generate(msgs, REPLY_TOKENS,
                                               prefill=" ".join(chosen[0].split()[:2])))
        return reply
    raise ValueError(variant)


# Replies with any knowledge_base.json leak, per variant (real /chat path only).
LEAKS = {}


def run(variants, splits=("dev", "test")):
    t0 = time.time()
    summary, rows = {}, []
    for split in splits:
        for variant in variants:
            counts = {"grounded": 0, "refusal": 0, "generic": 0}
            by_kind = {"shared": [0, 0], "personal": [0, 0]}
            for npc, q, pattern, kind in questions(split):
                reply = answer(npc, q, variant)
                if re.search(pattern, reply, re.IGNORECASE):
                    verdict = "grounded"
                elif REFUSAL.search(reply):
                    verdict = "refusal"
                else:
                    verdict = "generic"
                counts[verdict] += 1
                by_kind[kind][0] += verdict == "grounded"
                by_kind[kind][1] += 1
                rows.append({"split": split, "variant": variant, "npc": npc, "kind": kind,
                             "question": q, "verdict": verdict, "reply": reply})
            n = sum(counts.values())
            summary["%s/%s" % (split, variant)] = dict(counts, of=n, by_kind=by_kind)
            print("%-4s %-10s grounded %2d/%d  generic %2d  refusal %2d   shared %d/%d  personal %d/%d  (%.0fs)"
                  % (split, variant, counts["grounded"], n, counts["generic"], counts["refusal"],
                     *by_kind["shared"], *by_kind["personal"], time.time() - t0), flush=True)
    return summary, rows


LATENCY_CONVO = ["hi, who are you?", "where's the canteen?", "how many lecturers work here?",
                 "what do you do all day?", "when does term start?", "where are the staff offices?",
                 "how long have you worked here?", "thanks, any tips for a new student?"]


def latency(rounds=2):
    """In-conversation generation time for three fact layouts, via /chat.

      block      all facts in the system prompt (constant prefix)
      sys_block  only the retrieved facts in the system prompt (prefix changes per turn)
      server     the dialogue pipeline: retrieved facts in the late turn, repairs on

    Three NPCs, an 8-turn conversation each, transcript carried as the game
    carries it. Turn 1 of each conversation is dropped (cold either way).
    Layouts alternate across `rounds` so drift in machine load hits all three.
    """
    import statistics
    out = {}
    for _ in range(rounds):
        for layout in ("block", "sys_block", "server"):
            times = out.setdefault(layout, [])
            for npc in ("Prof. Adeyemi", "Officer Reyes", "Nadia"):
                history = []
                for i, q in enumerate(LATENCY_CONVO):
                    req = build_request(npc, q)
                    req.history = [gs.ChatTurn(**t) for t in history]
                    if layout == "sys_block":
                        chosen = retrieved(npc, req)
                        if chosen:
                            req.facts = "\n".join(chosen)
                    reply = chat_with(CONFIGS["server"] if layout == "server" else LEGACY, req)
                    if i:
                        times.append(reply.generation_ms)
                    history += [{"role": "user", "content": q},
                                {"role": "assistant", "content": reply.response}]
    summary = {k: {"p50": round(statistics.median(v), 1), "mean": round(statistics.mean(v), 1),
                   "p90": round(sorted(v)[int(len(v) * 0.9)], 1), "n": len(v)}
               for k, v in out.items()}
    for k, v in summary.items():
        print("%-10s median %6.0f ms  mean %6.0f ms  p90 %6.0f ms  n=%d"
              % (k, v["p50"], v["mean"], v["p90"], v["n"]), flush=True)
    return summary


if __name__ == "__main__":
    gs.load_scoring()  # chat() scores every reply; needs the knowledge base loaded
    assert gs.KNOWLEDGE_ITEMS, "KBD knowledge base did not load"
    args = sys.argv[1:]
    if args == ["latency"]:
        lat_path = REPO_ROOT / "evaluation" / "results" / "grounding_latency.json"
        lat_path.write_text(json.dumps(latency(), indent=1), encoding="utf-8")
        print("wrote %s" % lat_path.relative_to(REPO_ROOT))
        raise SystemExit(0)
    splits = ("dev", "test")
    if args and args[0] in ("dev", "test"):
        splits, args = (args[0],), args[1:]
    variants = args or ["block", "server"]
    summary, rows = run(variants, splits)
    summary["kbd_leaking_replies"] = LEAKS
    print("KB-leaking replies: %s" % LEAKS)
    # Only the default comparison writes the main results file; any other run
    # gets its own, so an ablation can never overwrite the reported numbers.
    out_path = OUT_PATH if (variants == ["block", "server"] and splits == ("dev", "test")) else \
        OUT_PATH.with_name("grounding_results_%s_%s.json" % ("-".join(splits), "-".join(variants)))
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print("wrote %s" % out_path.relative_to(REPO_ROOT))

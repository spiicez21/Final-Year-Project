"""One player message -> one NPC reply. Runs the stages in order; owns no rules.

    understand  intent.classify (learned), text.normalize for what the model reads
    learn       extractor.py (a fine-tuned span model) + memory.merge, before
                generation, so an introduction is usable in the reply to it
    retrieve    knowledge.select
    compose     composer.compose
    generate    the injected Generator
    check       guard.clean, guard.detect -> at most one regeneration -> guard.finish

The generator is injected so the pipeline can be driven by llama.cpp in the
server and by a scripted fake in unit tests, with no model loaded.
"""

from dataclasses import dataclass, field
from typing import Protocol

from . import composer, events, guard, intent as intents, knowledge, memory as player_memory
from . import extractor as facts_extractor
from .persona import eval_messages
from .text import normalize, similar


class Generator(Protocol):
    def __call__(self, messages: list, max_tokens: int, prefill: str = "",
                 repeat_penalty: float = 1.0) -> str:
        """Raw model text for the assistant turn. With `prefill`, the reply is
        forced to start with it and the returned text includes it."""


@dataclass
class Persona:
    name: str
    occupation: str = ""
    situation: str = ""
    intro: str = ""
    job_line: str = ""
    background: str = ""
    others: str = ""
    event_line: str = ""


@dataclass
class TurnInput:
    archetype: str
    message: str
    max_tokens: int = 40
    persona: Persona | None = None      # None -> the evaluation prompt, untouched
    facts: list = field(default_factory=list)
    fact_demos: list = field(default_factory=list)
    history: list = field(default_factory=list)
    player_memory: dict = field(default_factory=dict)
    # News this NPC has heard, and news it has not (the latter only to catch
    # leaks; it never enters the prompt). Shapes in events.py.
    known_events: list = field(default_factory=list)
    unheard_events: list = field(default_factory=list)


@dataclass
class TurnConfig:
    """Switches for ablations; the defaults are what the game gets."""
    normalize: bool = True           # expand chat shorthand before anything reads it
    fact_retrieval: bool = True      # False: all facts in the system prompt, none late
    memory_turn: str = "intent"      # "intent" | "always" | "never"
    repairs: bool = True             # guard regeneration and trimming
    deloop: bool = True              # drop repeated exchanges from the replayed transcript
    speech_shape: bool = True        # flatten lists, cap sentences (guard.clean)
    max_history_turns: int = 8
    learn_facts: bool = True         # run the player-fact extractor (extractor.py)
    news: bool = True                # learn reported events; share and retrieve heard ones


@dataclass
class TurnResult:
    response: str
    memory_updates: dict = field(default_factory=dict)
    player_memory: dict = field(default_factory=dict)
    facts_used: list = field(default_factory=list)
    intent: str = ""
    problems: list = field(default_factory=list)   # detected on the first reply
    repairs: list = field(default_factory=list)    # what was done about them
    generations: int = 1
    reported_event: dict | None = None             # the player just reported this
    shared_event_id: str = ""                      # this reply passed on that heard event
    event_leaks: list = field(default_factory=list)  # ids of unheard events the reply mentions


def run_turn(inp: TurnInput, generate: Generator, config: TurnConfig = TurnConfig(),
             extract=None) -> TurnResult:
    """`extract(message, context) -> (facts, event)`, both {slot: (value, score)},
    defaults to the GLiNER extractor's read(); tests pass a fake so no model is
    needed. A fake may also return only the facts dict."""
    if inp.persona is None:
        raw = generate(eval_messages(inp.archetype, inp.message), inp.max_tokens)
        return TurnResult(response=guard.clean_basic(raw))

    text = normalize(inp.message) if config.normalize else inp.message
    # The learned classifier reads the line as typed (it was trained on casual
    # typing); the pattern fallback normalises for itself.
    intent = intents.classify(inp.message)

    # Learn before generating, so an introduction can be answered by name. The
    # NPC's last line is context: "yuga" after "What's your name?" is a name.
    # The extractor reads the message as typed -- it was trained on casual
    # typing, and normalising first would hide the casing it uses for names.
    readings, event_reading = {}, {}
    if config.learn_facts:
        last_npc_line = next((t.get("content", "") for t in reversed(inp.history)
                              if t.get("role") == "assistant"), "")
        out = (extract or facts_extractor.default.read)(inp.message, last_npc_line)
        readings, event_reading = out if isinstance(out, tuple) else (out, {})
    memory, updates = player_memory.merge(inp.player_memory, readings)
    reported = (events.event_from(event_reading, inp.message, intent.kind == intents.REPORT)
                if config.news else None)
    if reported is None and config.news and intent.kind == intents.STATEMENT and not updates:
        # Anything else the player states and the memory did not take: kept as
        # a note in the player's own words, so an NPC can pass it on later.
        reported = events.note_from(inp.message)

    # Facts answer questions. Retrieving for statements put the head of
    # department's salary into his reply to "i'm in 2nd year ece".
    # News the NPC has heard joins the fact pool, so "is anything happening?"
    # retrieves it the same way "where's the canteen?" retrieves the floor.
    known_news = [events.sentence(e) for e in inp.known_events] if config.news else []
    # The other guests too, so "where's the officer?" has an answer to find.
    guests = knowledge.guest_facts(inp.persona.others)
    facts_used = []
    if (config.fact_retrieval and (inp.facts or known_news or guests)
            and intent.kind in (intents.QUESTION, intents.ABOUT_NPC)):
        facts_used = knowledge.select(text, knowledge.fact_pool(inp.facts, inp.persona.job_line)
                                      + known_news + guests,
                                      knowledge.split_sentences(inp.persona.background))
    # A greeted NPC passes on the freshest news it heard from someone else.
    # Incidents only: an NPC volunteers something urgent, not every remark it
    # was told, which would make every greeting a monologue.
    rumour = None
    if config.news and intent.kind == intents.GREETING:
        rumour = next((e for e in reversed(inp.known_events)
                       if e.get("fresh") and e.get("heard_from") not in ("", "player")
                       and e.get("kind", "incident") != "note"), None)

    # The event this reply is expected to carry: the rumour it was primed with,
    # or heard news that retrieval picked for the question.
    news = rumour or next((e for e in inp.known_events if events.sentence(e) in facts_used), None)

    messages = composer.compose(inp, text, intent, memory, facts_used, config, rumour)
    shape = guard.clean if config.speech_shape else guard.clean_basic
    reply = shape(generate(messages, inp.max_tokens))
    generations = 1

    p = inp.persona
    check = guard.Check(text=text, intent=intent, memory=memory,
                        history=composer.transcript(inp.history, config.max_history_turns, config),
                        facts_used=facts_used, npc_name=p.name, job_line=p.job_line,
                        persona_text=" ".join([p.name, inp.archetype, p.occupation, p.intro, p.job_line]),
                        updates=updates, news=news)
    problems = guard.detect(reply, check)
    repairs = []
    if config.repairs:
        regen = guard.plan_regeneration(reply, problems, check)
        if regen:
            retry_messages = messages
            if regen.hide_reply:
                hidden = [dict(t) for t in inp.history]
                keep = []
                for i in range(0, len(hidden) - 1, 2):
                    if not similar(hidden[i + 1].get("content", ""), regen.hide_reply):
                        keep += hidden[i:i + 2]
                retry_input = TurnInput(**{**inp.__dict__, "history": keep})
                retry_messages = composer.compose(retry_input, text, intent, memory, facts_used, config)
            retry = shape(generate(retry_messages, inp.max_tokens, prefill=regen.prefill))
            generations += 1
            retry, fallback = guard.after_regeneration(retry, regen, check)
            if guard.worse(reply, retry, check):
                # Never trade a flawed reply for a wrong one.
                repairs.append(regen.name + "_rejected")
            else:
                reply = retry
                repairs.append(regen.name)
                if fallback:
                    repairs.append(fallback)
        reply, trimmed = guard.finish(reply, check)
        repairs += trimmed

    shared = ""
    if config.news:
        shared = next((e.get("id", "") for e in inp.known_events if e.get("fresh")
                       and events.mentions(reply, e)), "")
    return TurnResult(response=reply, memory_updates=updates, player_memory=memory,
                      facts_used=facts_used, intent=intent.kind, problems=sorted(problems),
                      repairs=repairs, generations=generations, reported_event=reported,
                      shared_event_id=shared,
                      event_leaks=events.leaks(reply, inp.unheard_events) if config.news else [])

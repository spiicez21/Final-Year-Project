"""The one place that decides what goes into the prompt.

Layout, in order:

    system      persona, background, other guests, [memory clause]
    priming     identity / job / background / event turns (persona.py)
    fact demos  the game's worked examples (pay, student numbers)
    transcript  recent exchanges, player turns normalised, loops removed
    late turn   AT MOST ONE just-had exchange, chosen by intent:
                  recall, or greeting a returning player -> memory
                  otherwise, if facts were retrieved     -> those facts
    message     the player's line, normalised

Why a single late turn. Everything before the transcript is identical from
turn to turn, so llama.cpp reuses it from cache; only the tail changes. And on
a 1.1B model the last exchange before the question dominates the reply (memory
placement 14/40 early vs 21/40 late). That power is why only one thing may sit
there: with the memory turn inserted before every message, returning players
asking "what do you do" got "I don't do anything" -- the memory crowded out the
NPC's own job.

Why loops are removed from the transcript. Greedy decoding copies what it can
see. Once "I don't know anything about that." was in the replayed transcript,
the next three replies were the same line. An exchange whose reply repeats an
earlier reply is dropped from what is replayed (the game still shows it).
"""

from . import events, knowledge, memory as player_memory
from .intent import GREETING, RECALL, REPORT
from .persona import (BACKGROUND_CLAUSE, FACTS_CLAUSE, MEMORY_CLAUSE, OTHERS_CLAUSE,
                      PERSONA_TEMPLATE, priming_turns)
from .text import normalize, similar


def system_prompt(inp, memory: dict, config) -> str:
    p = inp.persona
    return PERSONA_TEMPLATE.format(
        name=p.name,
        archetype=inp.archetype,
        occupation=p.occupation or f"a {inp.archetype}",
        situation=p.situation or "a college department open day",
        background=BACKGROUND_CLAUSE.format(background=p.background.strip()) if p.background else "",
        # Naming the other guests lets an NPC hand a question on ("you'd want
        # to ask Ms. Okafor") instead of inventing an answer.
        others=OTHERS_CLAUSE.format(others=p.others.strip()) if p.others else "",
        facts=FACTS_CLAUSE.format(facts="\n".join(inp.facts))
              if inp.facts and not config.fact_retrieval else "",
        memory=MEMORY_CLAUSE.format(memory=player_memory.render(memory)) if memory else "",
    )


def transcript(history: list, max_turns: int, config) -> list:
    """Exchanges to replay: normalised, loops dropped, opening exchange kept.

    The opening exchange usually set the subject, so it survives truncation;
    the window slides over the rest.
    """
    exchanges, pending = [], None
    for turn in history:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role == "user":
            pending = normalize(content) if config.normalize else content
        elif role == "assistant" and pending is not None:
            exchanges.append((pending, content))
            pending = None
    if config.deloop:
        kept = []
        for user, reply in exchanges:
            if not any(similar(reply, r) for _, r in kept):
                kept.append((user, reply))
        exchanges = kept
    turns = []
    for user, reply in exchanges:
        turns += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    if len(turns) > max_turns:
        turns = turns[:2] + turns[-(max_turns - 2):]
    return turns


def late_turn(text: str, intent, memory: dict, facts_used: list, has_history: bool, config,
              rumour: dict | None = None) -> list:
    """The single just-had exchange placed before the message (see module doc).

    Priority: the player's memory for recall; how to take a report; news to pass
    on when greeted; otherwise the facts the question needs.
    """
    mode = config.memory_turn
    show_memory = memory and (
        mode == "always"
        or (mode == "intent" and (intent.kind == RECALL
                                  or (intent.kind == GREETING and memory.get("name")
                                      and not has_history))))
    if show_memory:
        return player_memory.demonstration(memory)
    if intent.kind == REPORT:
        return events.report_demonstration()
    if rumour and intent.kind == GREETING:
        return events.rumour_demonstration(rumour)
    if facts_used:
        return knowledge.demonstration(text, facts_used)
    return []


def compose(inp, text: str, intent, memory: dict, facts_used: list, config,
            rumour: dict | None = None) -> list:
    p = inp.persona
    occupation = p.occupation or f"a {inp.archetype}"
    messages = [{"role": "system", "content": system_prompt(inp, memory, config)}]
    messages += priming_turns(p.name, occupation,
                              p.intro or "I'm here for the open day, meeting students.",
                              p.job_line or "", p.background or "", p.event_line or "",
                              p.situation or "")
    messages += [t for t in inp.fact_demos if t.get("role") in ("user", "assistant")]
    history = transcript(inp.history, config.max_history_turns, config)
    messages += history
    messages += late_turn(text, intent, memory, facts_used, bool(history), config, rumour)
    messages.append({"role": "user", "content": text})
    return messages

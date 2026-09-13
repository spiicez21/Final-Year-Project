"""One player message -> one NPC reply. Runs the stages in order; owns no rules.

    understand  text.normalize, intent.classify
    learn       memory.extract + merge (before generation, so an introduction
                is usable in the reply to that same introduction)
    retrieve    knowledge.select
    compose     composer.compose
    generate    the injected Generator
    check       guard.clean, guard.detect -> at most one regeneration -> guard.finish

The generator is injected so the pipeline can be driven by llama.cpp in the
server and by a scripted fake in unit tests, with no model loaded.
"""

from dataclasses import dataclass, field
from typing import Protocol

from . import composer, guard, intent as intents, knowledge, memory as player_memory
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


def run_turn(inp: TurnInput, generate: Generator, config: TurnConfig = TurnConfig()) -> TurnResult:
    if inp.persona is None:
        raw = generate(eval_messages(inp.archetype, inp.message), inp.max_tokens)
        return TurnResult(response=guard.clean_basic(raw))

    text = normalize(inp.message) if config.normalize else inp.message
    intent = intents.classify(text)

    updates = player_memory.extract(text)
    memory = player_memory.merge(inp.player_memory, updates)

    # Facts answer questions. Retrieving for statements put the head of
    # department's salary into his reply to "i'm in 2nd year ece".
    facts_used = []
    if config.fact_retrieval and inp.facts and intent.kind in (intents.QUESTION, intents.ABOUT_NPC):
        facts_used = knowledge.select(text, knowledge.fact_pool(inp.facts, inp.persona.job_line),
                                      knowledge.split_sentences(inp.persona.background))

    messages = composer.compose(inp, text, intent, memory, facts_used, config)
    shape = guard.clean if config.speech_shape else guard.clean_basic
    reply = shape(generate(messages, inp.max_tokens))
    generations = 1

    p = inp.persona
    check = guard.Check(text=text, intent=intent, memory=memory,
                        history=composer.transcript(inp.history, config.max_history_turns, config),
                        facts_used=facts_used, npc_name=p.name, job_line=p.job_line,
                        persona_text=" ".join([p.name, inp.archetype, p.occupation, p.intro, p.job_line]),
                        updates=updates)
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
            reply = shape(generate(retry_messages, inp.max_tokens, prefill=regen.prefill))
            generations += 1
            repairs.append(regen.name)
            reply, fallback = guard.after_regeneration(reply, regen, check)
            if fallback:
                repairs.append(fallback)
        reply, trimmed = guard.finish(reply, check)
        repairs += trimmed

    return TurnResult(response=reply, memory_updates=updates, player_memory=memory,
                      facts_used=facts_used, intent=intent.kind, problems=sorted(problems),
                      repairs=repairs, generations=generations)

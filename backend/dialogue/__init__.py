"""The in-character dialogue pipeline behind /chat.

One player message becomes one NPC reply through fixed stages, each in its own
module, with a single place deciding what goes into the prompt:

    understand   text.py, intent.py   normalise chat shorthand; what is being asked
    learn        memory.py            facts the player stated about themselves
    retrieve     knowledge.py         the NPC facts this message needs
    compose      persona.py,          stable persona prefix + transcript + ONE
                 composer.py          late context turn chosen by intent
    generate     (injected)           llama.cpp in the server, a fake in tests
    check        guard.py             clean the reply; detect and repair
                                      refusals, repeats, invented names,
                                      questions already answered

turn.py runs the stages. See Docs/NPC_DIALOGUE_ARCHITECTURE.md for why the
pipeline is shaped this way and what each stage was measured to fix.

The evaluation prompt (no persona) does not go through any of this: it is
persona.eval_messages(), byte-identical to evaluation/run_*.py, so /chat
without a persona still reproduces the distribution the paper's numbers were
computed over.
"""

from .turn import Generator, Persona, TurnConfig, TurnInput, TurnResult, run_turn

__all__ = ["Generator", "Persona", "TurnConfig", "TurnInput", "TurnResult", "run_turn"]

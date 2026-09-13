"""Who the NPC is: the prompt templates and the priming turns.

Moved verbatim from gguf_server.py; the comments record why each piece is
shaped the way it is, and each was measured in play.
"""

import re

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                   "voice consistent with your role. Never break character.")

# In-character prompt used when the caller supplies a persona (the game does;
# the evaluation scripts do not).
#
# This is DELIBERATELY not the template above. SYSTEM_TEMPLATE is kept
# byte-identical to the one in evaluation/run_*.py so a /chat call with no
# persona reproduces the paper's prompt exactly. The moment a persona is
# supplied we are no longer sampling that distribution, so drift/KBD numbers
# from persona calls are NOT comparable to the reported results — they
# describe the in-game configuration instead. Both paths are kept so the
# comparison stays available rather than being quietly replaced.
#
# Kept short on purpose — see the note on priming_turns() below for why the
# persona is established by demonstration rather than by a longer instruction
# block. Residual assistant-prior leakage ("Sure, here's a sample response:")
# is stripped by guard.clean(); that leakage is itself an instance of the
# Instruct-prior-vs-persona competition the project documents as a drift
# mechanism.
PERSONA_TEMPLATE = (
    "You are {name}, a {archetype} — {occupation}. You are at {situation}.{background}{others}"
    "{facts}{memory} "
    "Reply in one or two short spoken sentences. Never break character."
)

# Optional clauses. Kept as separate fragments so an NPC with no background,
# or an event with nobody else at it, produces a prompt with no dangling
# sentence rather than an empty gap.
BACKGROUND_CLAUSE = " {background}"
OTHERS_CLAUSE = " Also here today: {others}."

# Game-world facts as one block in the system prompt. Not used by default:
# with every fact in the block, NPCs mixed facts up (31/50 correct) and the
# composer now shows only the retrieved facts, late (see knowledge.py). Kept
# for TurnConfig(facts_in_system=True), which reproduces the old layout.
FACTS_CLAUSE = "\nThings you know:\n{facts}\n"

# What this NPC has learned about the player (see memory.py), phrased about
# "the student you are talking to" so the model does not attribute the
# player's interests to itself.
MEMORY_CLAUSE = "\nWhat you remember about the student you are talking to: {memory}\n"


## Why the persona is taught by example rather than by instruction.
##
## The obvious approach — a long system prompt listing rules ("always answer
## questions about who you are", "never mention being an AI") — was tried and
## measurably backfired on this model. TinyLlama-1.1B answered "Who are you?"
## with "I'm not allowed to tell you that." and "What is your job like?" with
## "I don't have a job, I'm just a machine.": a flat persona break produced by
## the very prompt meant to prevent one.
##
## Two reasons. The adapters were fine-tuned against the short SYSTEM_TEMPLATE
## above, so a long structured instruction block is out-of-distribution for
## them; and a 1.1B model follows demonstrations far more reliably than
## negative rules, which mostly serve to put the forbidden words in context.
##
## So the system prompt stays short and close to the training distribution,
## and identity is established by seeding the conversation with two turns the
## NPC has already answered correctly. The model continues a pattern it can
## see instead of obeying rules it cannot hold.
def priming_turns(name: str, occupation: str, intro: str, job_line: str,
                   background: str = "", event_line: str = "",
                   situation: str = "") -> list:
    turns = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": f"I'm {name}, {occupation}."},
        {"role": "user", "content": "What are you doing here today?"},
        {"role": "assistant", "content": intro},
    ]
    # The third demonstration is not decorative. Several adapters were trained
    # with hand-authored refusal examples, and without a worked example of a
    # *permitted* question about their own work they generalise the refusal to
    # it — the executive answered "What is your job actually like?" with "I'm
    # not allowed to talk about my job." Showing one answered job question
    # scopes the refusal back to genuinely out-of-bounds topics.
    if job_line:
        turns += [
            {"role": "user", "content": "What is your job actually like?"},
            {"role": "assistant", "content": job_line},
        ]
        # A second demonstration in a casual, unpunctuated register. One
        # formal example was not enough: the executive answered "What is your
        # job actually like?" fine but still refused "whats your work like"
        # and "r job like" (2 of 6 phrasings tested). The refusal training in
        # some adapters keys partly on register, so the demonstrations have to
        # cover more than one. The short answer is the first sentence of
        # job_line, so the two examples agree on the facts while differing in
        # length and formality.
        first_sentence = re.split(r"(?<=[.!?])\s+", job_line.strip())[0]
        turns += [
            {"role": "user", "content": "whats your work like"},
            {"role": "assistant", "content": first_sentence},
        ]
    # Background as a demonstration too, not only as a line in the system
    # prompt. Stated in the system prompt alone it was mostly ignored: the
    # professor whose background says "came back to teach after four years in
    # industry" answered "Have you always worked here?" with "about a year",
    # and "Did you work outside academia?" with "No, I'm not allowed to work
    # outside academia" — flatly contradicting it. Same lesson as the identity
    # turns: on this model, shown beats told.
    if background:
        turns += [
            {"role": "user", "content": "How did you end up here?"},
            {"role": "assistant", "content": background},
        ]
    # "What is happening here?" is about the *event*, not the person, and
    # without a demonstration for it the adapter answers from its own training
    # topic instead: the police officer replied "we have received reports of a
    # group of people breaking into the computer science department", inventing
    # an incident at what is supposed to be an open day. (KBD scored null on
    # that reply — it matched no knowledge_base fact, so it was a hallucination
    # shaped by the archetype's topic prior, not a visibility-set leak.)
    if event_line:
        turns += [
            {"role": "user", "content": "what is happening here"},
            {"role": "assistant", "content": event_line},
        ]
        # Second phrasing, asking for the *reason* rather than the scene. One
        # demonstration covered "what is happening here" but left "Why is
        # everyone here?" answered with "I don't know" — and the host of the
        # event answering "I've never seen this before" is worse than a bland
        # line. Same register-coverage lesson as the job questions.
        if situation:
            turns += [
                {"role": "user", "content": "why is everyone here"},
                {"role": "assistant", "content": f"Everyone's here for {situation}."},
            ]
    return turns


def eval_messages(archetype: str, message: str) -> list:
    """The evaluation prompt, byte-identical to evaluation/run_*.py."""
    return [{"role": "system", "content": SYSTEM_TEMPLATE.format(archetype=archetype)},
            {"role": "user", "content": message}]

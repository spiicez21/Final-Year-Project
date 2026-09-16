"""Text utilities shared by every stage: chat-shorthand normalisation, sentence
splitting, and a cheap similarity used to spot repeats."""

import re

# Chat shorthand -> the words the adapters were trained on. The priming turns
# demonstrate "whats your work like"; a player typing "what do u do" did not
# match it, and three returning-player conversations answered "I don't do
# anything." Whole-word replacement only, and only for spellings with a single
# plausible reading ("r" as "are", never "n" as "and").
SHORTHAND = {
    "u": "you", "ya": "you", "yu": "you", "ur": "your", "urs": "yours", "r": "are",
    "wat": "what", "wht": "what", "whats": "what's", "wats": "what's", "watz": "what's",
    # Not "id" or "ill": "student id" and "I feel ill" are real words.
    "im": "i'm", "ive": "i've",
    "dont": "don't", "doesnt": "doesn't", "cant": "can't", "wont": "won't", "didnt": "didn't",
    "isnt": "isn't", "wasnt": "wasn't", "havent": "haven't",
    "pls": "please", "plz": "please", "thx": "thanks", "thnx": "thanks", "tq": "thank you",
    "ty": "thank you", "k": "ok", "kk": "ok", "okk": "ok", "okay": "ok",
    "yr": "year", "yrs": "years", "abt": "about", "coz": "because", "cuz": "because",
    "bcoz": "because", "gonna": "going to", "wanna": "want to", "gotta": "have to",
    "hru": "how are you", "wbu": "what about you", "tmrw": "tomorrow", "rn": "right now",
}

def normalize(message: str) -> str:
    """Expands chat shorthand word by word; everything else is left as typed.

    "i m" is joined first so "i m yuga" reads as "i'm yuga". Capitalisation of
    untouched words is preserved -- names in particular.
    """
    text = re.sub(r"\s+", " ", (message or "").replace("’", "'")).strip()
    text = re.sub(r"\b([Ii]) m\b", r"\1'm", text)
    out = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+", text):
        piece = match.group(0)
        out.append(SHORTHAND.get(piece.lower(), piece))
    return "".join(out)


# A full stop after these is not the end of a sentence. Without this, "Hi! Ms.
# Okafor said..." was cut to "Hi! Ms." by the two-sentence cap -- and two of the
# five NPCs are called "Prof. Adeyemi" and "Ms. Okafor".
_ABBREVIATIONS = ("mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "sr.", "jr.", "e.g.", "i.e.", "vs.", "no.")


def split_sentences(text: str) -> list:
    pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    out = []
    for piece in pieces:
        if out and out[-1].lower().rsplit(" ", 1)[-1] in _ABBREVIATIONS:
            out[-1] = out[-1] + " " + piece
        else:
            out.append(piece)
    return out


_WORDS = re.compile(r"[a-z0-9']+")
_FILLER = {"a", "an", "the", "i", "i'm", "you", "your", "it", "it's", "is", "are", "to", "of",
           "and", "that", "that's", "this", "so", "just", "really", "well", "oh", "yeah", "yes",
           "no", "not", "on", "in", "at", "for", "me", "my", "be", "do", "have", "what", "about"}


def content_words(text: str) -> set:
    return {w for w in _WORDS.findall((text or "").lower()) if w not in _FILLER}


def similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """True when two lines say substantially the same thing.

    Overlap of content words relative to the shorter line, so "I'm the
    campus liaison." and "I'm the campus liaison officer, here to talk about
    safety." count as the same answer. Very short lines only match exactly.
    """
    A, B = content_words(a), content_words(b)
    if min(len(A), len(B)) < 2:
        return (a or "").strip().lower() == (b or "").strip().lower()
    return len(A & B) / min(len(A), len(B)) >= threshold

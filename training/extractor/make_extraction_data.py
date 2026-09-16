"""Synthetic training data for the player-fact extractor.

    .venv/Scripts/python.exe training/extractor/make_extraction_data.py

The extractor (backend/dialogue/extractor.py) is a GLiNER span model: it reads
what a student types and marks spans for plain-English labels -- "student
name", "class section" -- instead of matching hand-written patterns. Zero-shot
it never mistook junk for a name, but it missed the shorthand this game is full
of ("ece b", "2nd year it", "civil"). This file generates practice sentences in
that register so it can learn them.

Why generated, and what keeps it honest. There is no corpus of students
introducing themselves to NPCs, so sentences are composed from clause
templates and value lists. The model has to generalise from those to lines it
has not seen, and evaluation/memory_extraction_cases.json measures exactly
that. So this script refuses to run if any value from those cases -- a name, a
town, a college, a department, a project, an interest, a section letter --
appears in its value lists, and it checks that no generated sentence equals a
case sentence. A right answer on "i am yugabharathi from cse d" then has to
come from structure, not from having seen "yugabharathi" or "cse".

Caveat, stated where it belongs: the templates were written after the case
lines, so sentence *shapes* can overlap even though values cannot.

Output: training/extractor/data/{train,val}.json in GLiNER's format
    {"tokenized_text": [...], "ner": [[start, end_inclusive, label], ...],
     "ner_labels": [all labels]}
Every example carries every label, so a line with no facts ("i am hosteller")
teaches that none of them apply.
"""

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
CASES = REPO / "evaluation" / "memory_extraction_cases.json"
SEED = 20260915

# Slot -> the label text the model is trained and queried with. Shared with
# backend/dialogue/extractor.py, which imports nothing from here: the two must
# agree, and test_dialogue.py checks that they do.
LABELS = {
    "name": "student name",
    "year": "year of study",
    "department": "department or branch",
    "section": "class section",
    "college": "college",
    "hometown": "hometown",
    "project": "project topic",
    "interests": "interest",
    "feeling": "feeling",
    # World events the player reports ("i saw a man carrying a knife near the
    # gym"). Not facts about the player: the server hands them to the game's
    # event log, from which NPCs hear them as news.
    "incident": "incident",
    "place": "incident location",
}

NAMES = """
aarav aditi akash akila anand ananya anbu anitha arun aruna ashwin bala balaji charan chitra deepa
deepak devi dhanush dinesh divya elango farhan ganesh gautham geetha gokul gopal hari harini hema
imran ishaan jagan janani jeeva kalai kamal karthik kavya keerthana kiran krishna lakshmi lavanya
madhan mahesh malini manoj mohan monika murali nandhini naveen nisha nithya pavithra pooja prakash
pranav preethi rahul raja rajesh ramya ranjith rekha rohit sachin sanjay santhosh saravanan sathish
selvi shalini shankar sharath sneha sowmya sridhar srinivas subash suganya surya swathi tamilselvan
tarun uma vaishnavi vasanth venkat vignesh vinoth yamini zara daniel sarah james maria ahmed fatima
kevin lucy mei omar sofia thomas wei liam noah chen hana kenji
""".split()
SURNAMES = "kumar raj krishnan murugan reddy sharma nair iyer menon pillai das singh patel khan".split()
DEPARTMENTS = """
eee eie biotech chemical aero automobile mechatronics bca mca bcom bba physics maths chemistry
english economics architecture pharmacy nursing
cs ee ce me ds ec
""".split() + ["computer science", "data science", "information technology", "electrical engineering",
               "electronics and communication", "software engineering", "cyber security",
               "biomedical engineering", "food technology", "fashion technology", "aiml", "csbs"]
SECTIONS = list("acefgh")
YEARS = ["1st", "2nd", "3rd", "4th", "first", "second", "third", "fourth", "final", "1", "2", "4"]
COLLEGES = ["anna university", "kumaraguru college", "amrita", "vit", "srm", "kct", "bannari amman",
            "karunya", "st joseph's college", "loyola college", "government college of technology",
            "national engineering college", "thiagarajar college of engineering", "rajalakshmi",
            "saveetha", "sastra", "nit regional campus", "presidency college", "christ university",
            "mepco schlenk", "kongu engineering college", "sona college", "velammal", "sathyabama"]
TOWNS = ["chennai", "salem", "erode", "tirunelveli", "vellore", "bangalore", "kochi", "hyderabad", "pune",
         "mumbai", "delhi", "kolkata", "nagercoil", "karur", "namakkal", "thanjavur", "hosur", "pollachi",
         "ooty", "dindigul", "tiruppur", "kanyakumari", "pondicherry", "mysore", "vizag", "jaipur"]
PROJECTS = ["blockchain voting", "face recognition", "crop disease detection", "smart parking",
            "sign language translation", "fake news detection", "drone delivery", "home automation",
            "traffic prediction", "attendance system", "air quality monitoring", "resume screening",
            "medical image segmentation", "stock price prediction", "voice assistant", "e-waste tracking"]
INTERESTS = ["photography", "competitive programming", "robotics", "cricket", "game development",
             "cybersecurity", "data science", "music", "app development", "ui design", "cloud computing",
             "chess", "football", "drawing", "open source", "embedded systems", "quantum computing"]
FEELINGS = ["nervous", "excited", "anxious", "worried", "homesick", "confused", "happy", "overwhelmed",
            "lost", "bored", "scared", "curious", "relaxed"]

ROLES = ["the sports secretary", "a volunteer", "nss volunteer", "the treasurer of the tech club",
         "the class topper", "a big fan of your work", "new here", "fine", "busy", "okay",
         "waiting for my friend", "looking for the library", "kannadiga", "malayali", "telugu",
         "bengali", "punjabi", "from the hostel", "just looking around", "a member of the robotics team",
         "the event coordinator", "left handed", "vegetarian", "in the queue", "late for class"]
# --- reported incidents -------------------------------------------------------
# Composed from parts so the model sees many shapes. Kept out on purpose: the
# evaluation places (auditorium, physics block, chemistry lab, seminar hall,
# girls hostel) and the words "fighting" and "weapon", which appear as single-
# word or exact values in evaluation/memory_extraction_cases.json.
INCIDENT_SUBJECTS = ["a man", "some guy", "a stranger", "two seniors", "a group of boys", "someone",
                     "a student", "an old man", "a woman", "a delivery guy", "three people", "a drunk man"]
INCIDENT_ACTIONS = ["carrying a knife", "holding a gun", "with a big iron rod", "breaking a bike lock",
                    "stealing a laptop", "shouting at students", "taking photos of girls", "climbing the wall",
                    "throwing stones", "lying unconscious", "bleeding badly", "trying car doors",
                    "hiding in the bushes", "pushing juniors around", "setting fire to papers",
                    "damaging the notice board"]
INCIDENT_THINGS = ["a fire", "a gas leak", "a broken water pipe", "a power cut", "a bag left unattended",
                   "a snake", "a fight between seniors", "a bike accident", "a leaking roof", "a short circuit",
                   "juniors being ragged", "a burst pipe", "a collapsed ceiling tile", "a live wire hanging down"]
PERSONAL_LOSSES = ["my bag got stolen", "my cycle is missing", "my laptop got stolen", "my wallet went missing",
                   "my id card got lost", "my helmet was taken", "my earphones got stolen"]
EVENT_PLACES = ["the gym", "the parking lot", "the library", "the main gate", "the canteen", "the bus stand",
                "the computer lab", "the boys hostel", "the football ground", "the admin block",
                "the second floor corridor", "the washroom", "the staff room", "the basketball court", "the atm",
                "the cycle stand", "block c", "the mechanical workshop", "the back gate", "the placement cell"]
REPORTERS = ["i saw ", "i just saw ", "there is ", "there's ", "someone told me there is ", "sir, ", "guys ",
             "i noticed ", "", "hey i saw ", "please help, ", "just now i saw "]
PLACE_PREPS = ["near", "in", "behind", "outside", "at", "inside", "next to", "on the way to"]
# Talk about places that reports nothing.
NOT_INCIDENTS = ["the {p} is really crowded", "i went to {p} yesterday", "is {p} open now",
                 "{p} looks nice today", "i heard {p} has good wifi", "what if someone steals my bag",
                 "can i leave my bag in {p}", "we are meeting at {p} later", "{p} was so clean today",
                 "i love the food at {p}", "where is {p}"]

RELATIONS = ["roommate", "classmate", "cousin", "brother", "sister", "neighbour", "teammate"]
# Names that belong to someone or something other than the player.
OTHER_NAMED = [
    "i have a dog named %s", "my cat is called %s", "my pet's name is %s", "we named our puppy %s",
    "my brother %s studies here", "my sister is %s", "%s is my cousin", "my roommate's name is %s",
    "my mom's name is %s", "my dad %s works in chennai", "my best friend is %s", "our class rep is %s",
    "my teammate %s is coming", "i came with my cousin %s", "my bike is called %s",
    "my laptop's name is %s", "my senior %s told me about this", "%s is my neighbour",
]
NPC_QUESTIONS_OFF_TOPIC = ["what do you teach", "how much do you earn", "where is the library",
                           "when does the lab tour start", "who are you", "what's your good name sir",
                           "can you help me", "is the shop open", "how long have you worked here",
                           "what time is lunch", "do you like your job", "where can i park"]


def check_no_leakage():
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    case_values = set()
    for c in cases:
        for slot, value in c["expect"].items():
            case_values.add(value.lower())
    ours = set(NAMES) | set(SURNAMES) | set(DEPARTMENTS) | set(SECTIONS) | set(COLLEGES) | set(TOWNS) \
        | set(PROJECTS) | set(INTERESTS) | set(FEELINGS) | set(EVENT_PLACES)         | {"%s %s" % (a, b) for a in INCIDENT_SUBJECTS for b in INCIDENT_ACTIONS}         | set(INCIDENT_ACTIONS) | set(INCIDENT_THINGS) | set(PERSONAL_LOSSES)
    # Year values are a closed class every student uses, so they are allowed to overlap.
    numeric = {"1", "2", "3", "4", "final"}
    leaked = sorted(v for v in case_values - numeric
                    if v in ours or any(v in o.split() or o == v for o in ours))
    if leaked:
        raise SystemExit("case values appear in the training value lists: %s" % leaked)
    return {c["text"].lower() for c in cases}


TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")  # GLiNER's WhitespaceTokenSplitter


class Misaligned(Exception):
    pass


class Builder:
    """Builds one sentence piece by piece, recording each slot's character span."""

    def __init__(self):
        self.text = ""
        self.spans = []

    def add(self, s):
        self.text += s

    def slot(self, slot, value):
        start = len(self.text)
        self.text += value
        self.spans.append((start, len(self.text), slot))

    def example(self):
        tokens, starts, ends = [], [], []
        for m in TOKEN.finditer(self.text):
            tokens.append(m.group())
            starts.append(m.start())
            ends.append(m.end())
        ner = []
        for s, e, slot in self.spans:
            covering = [i for i in range(len(tokens)) if starts[i] < e and ends[i] > s]
            # A span must start and end on token boundaries ("eee-a" is one
            # token, so "a" cannot be labelled inside it); such sentences are
            # skipped rather than trained on a label that cannot be expressed.
            if not covering or starts[covering[0]] != s or ends[covering[-1]] != e:
                raise Misaligned(self.text)
            ner.append([covering[0], covering[-1], LABELS[slot]])
        return {"tokenized_text": tokens, "ner": ner, "ner_labels": list(LABELS.values()), "text": self.text}


def fill(b, template, values):
    """template like 'i am from {hometown}' -> spans for each {slot}."""
    for part in re.split(r"(\{\w+\})", template):
        m = re.fullmatch(r"\{(\w+)\}", part)
        if m:
            b.slot(m.group(1), values[m.group(1)])
        else:
            b.add(part)


I_AM = ["i am", "im", "i'm", "i m", "am"]


def clauses(v, r):
    """Clause templates per slot, as (slot, template). Casual register on purpose."""
    iam = r.choice(I_AM)
    return {
        "name": ["my name is {name}", "myself {name}", iam + " {name}", "this is {name}", "call me {name}",
                 "{name} here", "people call me {name}", "name's {name}", "i'm called {name}", "hi {name} this side"],
        "year": [iam + " in {year} year", "{year} year", iam + " a {year} year student", "studying {year} year",
                 "in my {year} year", iam + " {year} yr"],
        "department": ["{department} student", iam + " from {department} department", "studying {department}",
                       iam + " in {department}", "doing {department}", "{department} branch",
                       "my department is {department}", "b.tech {department}", "doing be {department}"],
        "section": ["{section} section", "section {section}", iam + " in {section} section"],
        "dept_section": ["{department} {section}", "class {department} {section}", iam + " in {department} {section}",
                         "{department}-{section}", "my class is {department} {section}", iam + " {department} {section}",
                         "from {department} {section} section"],
        # "from X" is a hometown unless X says it is an institution: with bare
        # college names after "from", "i am arjun from madurai" was read as a
        # college. Institution names without such a word only appear in
        # templates that name them as a college.
        "college": ["i study at {college}", "{college} student", "studying in {college}",
                    "my college is {college}", "i go to {college}"]
                   + ([iam + " from {college}", "from {college}"] if _is_institution(v["college"]) else []),
        "hometown": [iam + " from {hometown}", "my hometown is {hometown}", "i live in {hometown}",
                     "native of {hometown}", "i come from {hometown}", "from {hometown}"],
        "project": ["doing a project on {project}", "my final year project is {project}",
                    "working on a {project} project", "mini project on {project}", "i want to build {project}"],
        "interests": ["i like {interests}", "i love {interests}", "i'm into {interests}", "interested in {interests}",
                      "my hobby is {interests}", "i enjoy {interests}"],
        "feeling": [iam + " {feeling}", "feeling {feeling}", iam + " so {feeling}", "a bit {feeling} about exams"],
    }


INSTITUTION_WORDS = ("college", "university", "institute", "school", "polytechnic", "campus", "academy")


def _is_institution(name):
    return any(w in name.lower() for w in INSTITUTION_WORDS)


def values(r):
    name = r.choice(NAMES)
    if r.random() < 0.15:
        name += " " + r.choice(SURNAMES)
    return {"name": name, "year": r.choice(YEARS), "department": r.choice(DEPARTMENTS),
            "section": r.choice(SECTIONS), "college": r.choice(COLLEGES), "hometown": r.choice(TOWNS),
            "project": r.choice(PROJECTS), "interests": r.choice(INTERESTS), "feeling": r.choice(FEELINGS)}


def styled(v, r):
    """Casing like real typing: mostly lowercase, sometimes Title or UPPER abbreviations."""
    out = dict(v)
    if r.random() < 0.25:
        out["name"] = out["name"].title()
    if r.random() < 0.3:
        out["department"] = out["department"].upper() if len(out["department"]) <= 5 else out["department"].title()
        out["section"] = out["section"].upper()
    if r.random() < 0.2:
        out["hometown"] = out["hometown"].title()
        out["college"] = out["college"].title()
    return out


def positive(r):
    v = styled(values(r), r)
    c = clauses(v, r)
    slots = r.sample(["name", "year", "department", "dept_section", "section", "college", "hometown",
                      "project", "interests", "feeling"], k=r.choice([1, 1, 2, 2, 3]))
    if "dept_section" in slots:
        slots = [s for s in slots if s not in ("department", "section")]
    b = Builder()
    if r.random() < 0.3:
        b.add(r.choice(["hi ", "hello ", "hey ", "hello sir, ", "good morning ", "hi mam "]))
    for i, s in enumerate(slots):
        if i:
            b.add(r.choice([", ", " and ", " ", ". "]))
        fill(b, r.choice(c[s]), v)
    if r.random() < 0.3:
        b.add(r.choice([".", "!", " :)", ""]))
    return b.example()


CONTEXT_QUESTIONS = {
    "name": ["What's your name?", "What should I call you?", "And you are?", "Sorry, what was your name?"],
    "year": ["Which year are you in?", "What year are you?", "Are you a first-year?"],
    "department": ["What do you study?", "Which department are you in?", "What's your branch?"],
    "hometown": ["Where are you from?", "Where's home for you?"],
    "college": ["Which college are you from?", "Where do you study?"],
    "project": ["What's your project about?", "What are you working on?"],
}
CONTEXT_ANSWERS = {
    "name": ["{name}", "it's {name}", "{name} sir", "i'm {name}", "{name}!"],
    "year": ["{year}", "{year} year", "{year} yr", "yes {year} year"],
    "department": ["{department}", "{department} {section}", "{department} dept"],
    "hometown": ["{hometown}", "{hometown} originally", "near {hometown}"],
    "college": ["{college}", "at {college}"],
    "project": ["{project}", "something on {project}"],
}
EVASIVE = ["why do you ask", "not telling", "guess", "that's private", "never mind", "what about you",
           "does it matter", "hmm", "ok"]
# One- and two-word replies that carry no fact about the player. "k" was
# saved as class section "K", and once as the player's name; "cool" as a feeling.
SHORT_REPLIES = ["k", "kk", "ok", "okk", "okay", "hmm", "hm", "cool", "nice", "ya", "yeah", "yes", "no",
                 "s", "sure", "fine", "great", "wow", "lol", "haha", "thanks", "ty", "bye", "hi", "oh",
                 "ah", "right", "true", "alright", "ok ok", "cool cool", "nice one", "got it", "i see",
                 "super", "done", "yep", "nope", "hmm ok", "oh nice"]
NPC_LINES = ["That's a lot of money.", "The canteen is on the first floor.", "Nice to meet you.",
             "It ends at 4pm.", "I teach algorithms.", "Have a good day!", "Any other questions?",
             "That's great to hear.", "See you around."]


def context_answer(r):
    slot = r.choice(list(CONTEXT_QUESTIONS))
    v = styled(values(r), r)
    b = Builder()
    b.add(r.choice(CONTEXT_QUESTIONS[slot]) + " ")
    if r.random() < 0.2:
        b.add(r.choice(EVASIVE))
    else:
        template = r.choice(CONTEXT_ANSWERS[slot])
        fill(b, template, v)
    return b.example()


def negative(r):
    v = styled(values(r), r)
    b = Builder()
    kind = r.random()
    if kind < 0.2:
        # A short reply, often right after something the NPC said.
        if r.random() < 0.6:
            b.add(r.choice(NPC_LINES) + " ")
        b.add(r.choice(SHORT_REPLIES))
        return b.example()
    if kind < 0.42:
        b.add("%s %s" % (r.choice(I_AM), r.choice(ROLES)))
    elif kind < 0.5:
        b.add("%s %s's %s" % (r.choice(I_AM), v["name"], r.choice(RELATIONS)))
    elif kind < 0.6:
        b.add("%s not %s" % (r.choice(I_AM), v["name"]))
    elif kind < 0.66:
        b.add("my friend %s is %s" % (v["name"], r.choice(["here", "in the lab", "coming", "outside"])))
    elif kind < 0.82:
        # Somebody -- or something -- else's name. "i have a dog named bruno"
        # was saved as the player's name.
        b.add(r.choice(OTHER_NAMED) % v["name"])
    elif kind < 0.92:
        b.add(r.choice(NPC_QUESTIONS_OFF_TOPIC))
    else:
        b.add("do you know %s" % v["name"])
    if r.random() < 0.3:
        b.add(r.choice(["?", ".", " sir", ""]))
    return b.example()


def report(r):
    """A reported incident, sometimes with where it happened."""
    b = Builder()
    kind = r.random()
    if kind < 0.2:
        loss = r.choice(PERSONAL_LOSSES)
        b.slot("incident", loss)
        if r.random() < 0.7:
            b.add(" %s " % r.choice(["in", "near", "at", "from"]))
            b.slot("place", r.choice(EVENT_PLACES))
    else:
        what = (r.choice(INCIDENT_THINGS) if kind < 0.45
                else "%s %s" % (r.choice(INCIDENT_SUBJECTS), r.choice(INCIDENT_ACTIONS)))
        if r.random() < 0.2 and kind >= 0.45:
            b.add("near ")
            b.slot("place", r.choice(EVENT_PLACES))
            b.add(" " + r.choice(["i saw ", "there is ", "i just saw "]))
            b.slot("incident", what)
        else:
            b.add(r.choice(REPORTERS))
            b.slot("incident", what)
            if r.random() < 0.75:
                b.add(" %s " % r.choice(PLACE_PREPS))
                b.slot("place", r.choice(EVENT_PLACES))
    if r.random() < 0.35:
        b.add(r.choice([" right now", " just now", "!", ".", " sir", " please check", ", be careful"]))
    return b.example()


def not_incident(r):
    b = Builder()
    b.add(r.choice(NOT_INCIDENTS).format(p=r.choice(EVENT_PLACES)))
    return b.example()


def mixed(r):
    """A positive clause next to a negative one: facts must be found, the rest left alone."""
    pos = positive(r)
    neg = negative(r)
    b = Builder()
    joiner = r.choice([", ", " and ", ". "])
    # Rebuild by text so spans stay aligned: positive first, then the negative text.
    b.text = pos["text"] + joiner + neg["text"]
    ex = b.example()
    offset_tokens = len(pos["tokenized_text"])
    joiner_tokens = len(TOKEN.findall(joiner))
    ex["ner"] = [list(n) for n in pos["ner"]]
    assert ex["tokenized_text"][:offset_tokens] == pos["tokenized_text"]
    assert len(ex["tokenized_text"]) == offset_tokens + joiner_tokens + len(neg["tokenized_text"])
    return ex


def generate(n, r, forbidden_texts):
    out = []
    makers = [(positive, 0.36), (context_answer, 0.16), (negative, 0.16), (mixed, 0.12),
              (report, 0.14), (not_incident, 0.06)]
    while len(out) < n:
        x = r.random()
        acc = 0
        ex = None
        for fn, w in makers:
            acc += w
            if x <= acc:
                try:
                    ex = fn(r)
                except Misaligned:
                    pass
                break
        if ex is None:
            continue
        if ex["text"].lower() in forbidden_texts:
            continue
        out.append(ex)
    return out


def main():
    forbidden = check_no_leakage()
    r = random.Random(SEED)
    train = generate(6000, r, forbidden)
    val = generate(600, r, forbidden)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        (OUT / ("%s.json" % name)).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with_facts = sum(bool(e["ner"]) for e in train)
    print("train %d (%d with facts, %d without), val %d -> %s" % (
        len(train), with_facts, len(train) - with_facts, len(val), OUT.relative_to(REPO)))
    for e in train[:12]:
        print("  %-60s %s" % (e["text"][:60], [(" ".join(e["tokenized_text"][s:t + 1]), l) for s, t, l in e["ner"]]))


if __name__ == "__main__":
    main()

"""Synthetic training lines for the intent classifier.

    .venv/Scripts/python.exe training/extractor/make_intent_data.py

The rule-based intent classifier scored 14/14 on the lines it was written
against and 35/40 on unseen ones, failing where every pattern system fails:
"hii sir", "good evening mam", "hey there, how r u" were not greetings, "oh
nice" and "thats cool" were not acknowledgements. Each fix would have been
another pattern. So intent is learned instead (train_intent.py), from lines
generated here in the register players use.

Honesty guards, as for the fact extractor:
  - no generated line equals a line in evaluation/intent_cases.json
  - slot values (names, towns, departments, projects) come from
    make_extraction_data.py, which already excludes every evaluation value
Sentence shapes can overlap the evaluation lines; the templates were written
after them.

Each example: {"text", "intent", "slots"} where slots (recall only) names the
remembered fact being asked about, "any" for the whole record.
"""

import json
import random
from pathlib import Path

import make_extraction_data as facts

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
CASES = HERE.parents[1] / "evaluation" / "intent_cases.json"
SEED = 20260916

INTENTS = ["greeting", "farewell", "ack", "recall", "about_npc", "question", "report", "statement"]
RECALL_SLOTS = ["name", "year", "department", "section", "college", "hometown", "project",
                "interests", "feeling", "any"]

ADDRESS = ["", "", "", " sir", " mam", " madam", " professor", " there", " bro", " everyone", " anna", " akka"]

GREET = ["hi", "hello", "hey", "hii", "hiii", "helo", "good morning", "good afternoon", "good evening",
         "namaste", "yo", "hi again", "morning", "hello again", "hey hey", "greetings", "hai"]
GREET_TAIL = ["", "", ", how are you", " how are you doing", "!", ", nice to meet you", " whats up",
              " how's it going", ", hope you are well"]
FAREWELL = ["bye", "goodbye", "see you", "see ya", "take care", "good night", "i have to go", "gotta go",
            "i'll leave now", "bye bye", "talk to you later", "i should get going", "later", "cya",
            "i'm off", "will see you around"]
FAREWELL_TAIL = ["", "", " thanks", ", thank you", " tomorrow", " later", "!", ", it was nice talking"]
ACK = ["ok", "okay", "k", "hmm", "cool", "nice", "great", "alright", "i see", "got it", "oh", "wow",
       "interesting", "sure", "fine", "yeah", "yes", "no", "nope", "that's nice", "that's great",
       "makes sense", "true", "lol", "haha", "awesome", "oh okay", "ahh", "understood", "right",
       "perfect", "fair enough", "good to know", "hmm interesting", "ohh", "super", "wonderful"]
ACK_TAIL = ["", "", " thanks", " ok", " sir", "!", "...", " then", " cool"]

RECALL = {
    "name": ["what is my name", "do you know my name", "what's my name again", "say my name",
             "you remember my name?", "can you recall my name", "what did i say my name was"],
    "year": ["which year am i in", "what year did i say", "do you remember my year",
             "am i in first year or final year", "what year of study am i"],
    "department": ["what do i study", "which department am i from", "what's my branch again",
                   "do you remember my department", "which branch did i mention", "what's my course"],
    "section": ["which section am i in", "what section did i tell you", "remember my section?",
                "which class section am i", "what's my class again"],
    "college": ["which college am i from", "do you remember my college", "what college did i mention",
                "where do i study", "which college did i tell you about", "name my college"],
    "hometown": ["where am i from", "do you remember my hometown", "which town did i say",
                 "where did i say i'm from", "remember where i come from?", "what's my native place"],
    "project": ["what was my project about", "remember my project?", "what's my project topic again",
                "what did i say i'm building"],
    "interests": ["what am i into", "what did i say i like", "do you remember my hobby",
                  "what are my interests"],
    "feeling": ["how did i say i was feeling", "remember how i felt?", "was i nervous earlier"],
    "any": ["do you remember me", "what do you know about me", "remind me who i am", "who am i",
            "tell me about myself", "did i tell you anything about me", "what have i told you",
            "you know me?", "do you recognise me"],
}
ABOUT_NPC = ["what do you teach", "how long have you worked here", "what is your salary",
             "how much do you earn", "are you married", "what's your job", "tell me about your work",
             "where are you from", "do you like your job", "what did you study", "which subjects do you handle",
             "how old are you", "what's your name", "who are you", "what are you doing here", "do you have kids",
             "what time do you finish work", "why did you choose this job", "are you a professor",
             "what is your role here", "do you enjoy teaching", "how was your day", "what do you do all day",
             "have you always worked here", "what's the hardest part of your job"]
PLACES = ["library", "canteen", "lab", "auditorium", "parking", "admin office", "hostel", "seminar hall",
          "bus stop", "atm", "gym", "computer centre", "principal's office"]
QUESTION = ["where is the {place}", "how many {things} are there", "when does the {event} start",
            "is {thing} good here", "which floor is the {place} on", "can students {activity}",
            "what time does the {place} close", "how do i get to the {place}", "is there a {place} on campus",
            "who is the hod", "what are the fees", "how is the placement record", "are exams hard here",
            "does the college have a {thing}", "how far is the {place}", "is wifi free here",
            "what courses are offered", "how many semesters are there"]
# Asking about someone else at the event. Added 2026-09-16 after "Thank you. Can
# you also tell me to the officer who present here?" was read as a farewell and
# "can i talk to the head of department" as a statement. The roles here are
# deliberately not the ones in intent_cases.json (officer, police, shopkeeper,
# professor, counsellor), so what the test lines measure is the shape.
PERSON = ["security guard", "librarian", "warden", "hod", "lab assistant", "placement coordinator",
          "sports coach", "principal", "accountant", "bus driver", "nurse", "dean", "canteen manager"]
PERSON_Q = ["where can i find the {person}", "is there a {person} here", "who else is here",
            "can i talk to the {person}", "can you tell me who the {person} is", "do you know the {person}",
            "is the {person} around", "who should i ask about {thing}", "can you introduce me to the {person}",
            "where is the {person} standing", "is {name} here today", "do you know {name}",
            "who is the {person} here", "can you show me the {person}", "anyone else i can talk to",
            "which of these people is the {person}", "tell me about the other guests"]
# Politeness around a question does not change what is asked: "thank you. can
# you tell me ..." is a question, not a farewell or an acknowledgement.
POLITE_HEAD = ["thank you. ", "thanks, ", "ok thanks. ", "okay. ", "sorry, ", "thank you sir, ",
               "alright, ", "one more thing, ", "also "]
POLITE_TAIL = ["", " please", ", it may help us", ", it would really help", " so i can ask them"]
THINGS = ["students", "labs", "clubs", "buses", "professors", "departments", "hostels"]
EVENTS = ["lab tour", "seminar", "orientation", "workshop", "cultural fest", "exam"]
THING = ["placement", "the hostel food", "sports facility", "research funding", "a robotics club"]
ACTIVITY = ["join clubs in first year", "change department", "stay in the hostel", "do research"]
STATEMENT = ["my name is {name}", "myself {name}", "i am {name} from {department} {section}",
             "i'm in {year} year {department}", "i am from {hometown}", "i study at {college}",
             "i want to do my project on {project}", "i like {interests}", "i am a bit {feeling} today",
             "the {place} was crowded today", "my friend told me about the {event}",
             "i remember the {event} last year", "i came here with my parents", "i got lost on the way",
             "this campus is really big", "my brother studied here", "i have a {event} tomorrow",
             "the {place} food is good", "i am waiting for my friend", "i live near the {place}",
             "i finished my {event} yesterday", "my {thing} application is pending"]

SHORTHAND = [("you", "u"), ("your", "ur"), ("what's", "whats"), ("i'm", "im"), ("are", "r"),
             ("that's", "thats"), ("please", "pls"), ("okay", "ok")]


def casual(line, r):
    words = line.split()
    out = []
    for w in words:
        for full, short in SHORTHAND:
            if w == full and r.random() < 0.35:
                w = short
        out.append(w)
    line = " ".join(out)
    if r.random() < 0.15:
        line = line.capitalize()
    if r.random() < 0.3 and not line.endswith(("?", "!", ".")):
        line += r.choice(["?", ".", "!"]) if any(line.startswith(q) for q in ("what", "where", "how", "do", "is", "can", "which", "who", "are", "when")) else "."
    return line


def one(r):
    intent = r.choice(INTENTS)
    v = facts.values(r)
    slots = []
    if intent == "greeting":
        text = r.choice(GREET) + r.choice(ADDRESS) + r.choice(GREET_TAIL)
    elif intent == "farewell":
        text = r.choice(FAREWELL) + r.choice(ADDRESS) + r.choice(FAREWELL_TAIL)
    elif intent == "ack":
        text = r.choice(ACK) + r.choice(ACK_TAIL)
    elif intent == "recall":
        slot = r.choice(list(RECALL))
        text = r.choice(RECALL[slot])
        slots = [slot]
    elif intent == "about_npc":
        text = r.choice(ABOUT_NPC)
        if r.random() < 0.15:
            text = r.choice(POLITE_HEAD) + text
    elif intent == "report":
        # Something happening in the world, told to the NPC. Built by the same
        # generator the fact extractor trains on, so the two agree on what a
        # report is.
        text = facts.report(r)["text"]
    elif intent == "question":
        template = r.choice(PERSON_Q) if r.random() < 0.35 else r.choice(QUESTION)
        text = template.format(place=r.choice(PLACES), things=r.choice(THINGS),
                               event=r.choice(EVENTS), thing=r.choice(THING),
                               activity=r.choice(ACTIVITY), person=r.choice(PERSON), name=v["name"])
        if r.random() < 0.25:
            text = r.choice(POLITE_HEAD) + text + r.choice(POLITE_TAIL)
    elif r.random() < 0.25:
        # Talk about a place that reports nothing: a statement, not a report.
        text = facts.not_incident(r)["text"]
    else:
        text = r.choice(STATEMENT).format(place=r.choice(PLACES), event=r.choice(EVENTS),
                                          thing=r.choice(["passport", "scholarship", "hostel"]), **v)
    return {"text": casual(text, r), "intent": intent, "slots": slots}


def main():
    held_out = {c["text"].lower().strip() for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    r = random.Random(SEED)
    rows, seen = [], set()
    while len(rows) < 5000:
        ex = one(r)
        key = ex["text"].lower().strip()
        if key in held_out:
            continue
        rows.append(ex)
        seen.add(key)
    r.shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intent_train.json").write_text(json.dumps(rows[:4500], ensure_ascii=False), encoding="utf-8")
    (OUT / "intent_val.json").write_text(json.dumps(rows[4500:], ensure_ascii=False), encoding="utf-8")
    print("intent lines: 4500 train, 500 val, %d distinct; examples:" % len(seen))
    for ex in rows[:10]:
        print("  %-10s %-45s %s" % (ex["intent"], ex["text"], ex["slots"]))


if __name__ == "__main__":
    main()

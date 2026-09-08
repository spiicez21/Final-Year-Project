extends RefCounted
class_name CampusFacts

## Game-world knowledge for the open day: department size, who teaches what,
## working conditions, pay.
##
## ---------------------------------------------------------------------------
## This is NOT data/processed/knowledge_base.json, and must never be merged
## into it.
##
## That file is the scoring artifact for Contribution C1: seven evaluation
## scripts read it, and its 39 signature phrases decide what counts as a
## factual reference when KBD is computed. Adding entries to it would change
## KBD for every banked result (alpha sweep, Conditions B and C, hybrid BC,
## calibration and recheck) and quietly invalidate them — and the modern-city
## dataset froze on 6 Sep 2026 anyway.
##
## These facts are deliberately lexically disjoint from those signature
## phrases (which are about Elm Street break-ins, insulin shipments, tenure
## cases, liquor distributors and so on), so putting them in an NPC's mouth
## does not trip the KBD scorer. In-game KBD keeps measuring exactly what it
## measured before.
## ---------------------------------------------------------------------------
##
## The adapters were never trained on any of this, so unlike the evaluation
## knowledge base there is no Condition B claim to protect here: injecting
## these into the prompt is the only way an NPC could know them, and doing so
## takes nothing away from the paper.
##
## All figures are invented for the demo. They are plausible for a UK
## university department, but they are not real institutional data.

## Known to everyone at the event. Kept to the things a visitor would actually
## ask about — a longer list costs prompt tokens on every single turn.
const SHARED := [
	"The department has about 480 undergraduates and 95 taught postgraduates.",
	"There are 34 academic staff and 11 professional services staff.",
	"The ground floor has the lecture rooms and the seminar hall.",
	"The first floor has the teaching labs and the canteen.",
	"The second floor has classrooms and the staff offices.",
	"Term runs from late September to mid-December, then January to late March.",
	"The open day runs from 10am to 4pm, with lab tours on the hour.",
]

## Private to one NPC: their own role, workload and pay. Keyed by the `name`
## in NpcDirector.SPAWNS.
##
## Pay is included because it is one of the first things people actually ask
## someone about their job, and an NPC that cannot answer it reads as hollow.
## Each NPC knows only their own — nobody is given anyone else's, so asking
## the shopkeeper what the head of department earns should not produce a
## number.
const PERSONAL := {
	"Prof. Adeyemi": [
		"I teach CS2011 Algorithms and Data Structures, and CS3040 Compilers.",
		"I supervise six final-year projects this year.",
		"I'm on grade 8, which is about 52,000 a year.",
		"I have about 18 contact hours a fortnight, and the rest is marking and research.",
	],
	"Halvorsen": [
		"I'm responsible for 34 academic staff and 11 support staff.",
		"I'm on grade 10 with a head-of-department allowance, so about 78,000 a year.",
		"I chair the staff-student committee and sit on the faculty board.",
		"I still teach CS1002 Introduction to Programming in the first semester.",
	],
	"Officer Reyes": [
		"I cover three campus sites, not just this one.",
		"I'm a police constable on the standard scale, about 41,000 a year.",
		"My shift is 7am to 3pm on weekdays, plus cover for events like today.",
		"Most of what I deal with is lost property and bike thefts.",
	],
	"Ms. Okafor": [
		"I usually have about 40 students on my caseload at any one time.",
		"I'm on grade 7, which is about 39,000 a year.",
		"I run drop-in sessions on Tuesdays and Thursdays, no appointment needed.",
		"Anything a student tells me stays between us unless somebody is at risk.",
	],
	"Nadia": [
		"I run the shop with two part-time staff, both students.",
		"We take about 1,800 a day in term time, and almost nothing over the summer.",
		"I open at 7am and close at 6pm, later during exams.",
		"The shop is mine, so what it makes after costs is what I earn.",
	],
}


## One worked example per NPC of answering "what do you earn". Facts in the
## system prompt are not enough on their own for pay: with the figure sitting
## right there in the prompt, the professor still answered "What do you earn?"
## with "I don't have that information" and the head of department with "I
## don't have one." Pay questions land squarely in the adapters' refusal
## training, and only a demonstration scopes it back.
const PAY_ANSWER := {
	"Prof. Adeyemi": "I'm on grade 8, so about 52,000 a year.",
	"Halvorsen": "Grade 10 with a head-of-department allowance, so about 78,000.",
	"Officer Reyes": "Standard constable scale, about 41,000 a year.",
	"Ms. Okafor": "I'm on grade 7, so about 39,000 a year.",
	"Nadia": "The shop's mine, so what it makes after costs is what I earn.",
}

## Demonstrations of answering questions *about* the facts, as alternating
## user/assistant turns. Two are enough to establish the shape: one numeric
## question about the department, one about the NPC's own pay.
static func demos_for(display_name: String) -> Array:
	var demos: Array = [
		{"role": "user", "content": "how many students are in the department"},
		{"role": "assistant",
		 "content": "About 480 undergraduates, and another 95 on taught masters."},
	]
	if PAY_ANSWER.has(display_name):
		# Two phrasings, same answer. One was not enough: with only "what do
		# you earn" demonstrated, "What is your salary?" still came back as
		# "I don't have a salary." Register coverage matters here exactly as
		# it did for the job questions.
		demos.append({"role": "user", "content": "what do you earn"})
		demos.append({"role": "assistant", "content": PAY_ANSWER[display_name]})
		demos.append({"role": "user", "content": "What is your salary?"})
		demos.append({"role": "assistant", "content": PAY_ANSWER[display_name]})
	return demos


## The facts block sent to the server for one NPC: shared campus knowledge
## plus that NPC's own. Returned as one newline-separated string.
static func for_npc(display_name: String) -> String:
	var lines: Array = []
	lines.append_array(SHARED)
	lines.append_array(PERSONAL.get(display_name, []))
	return "\n".join(lines)

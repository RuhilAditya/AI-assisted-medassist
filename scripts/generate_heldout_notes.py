"""Generate a HELD-OUT set of synthetic clinical notes with gold labels.

Why: the Phase 1 de-ID/NER numbers came from 6 hand-written notes that were also used
while tuning the rules, so they are optimistic. This generator uses different templates,
name lists, ID formats and phrasing, was written after the rules were frozen, and its
output is never used for tuning. All people, IDs and contacts are fictional.

Gold per note: phi ({type, value} strings that must be removed), keep (clinical strings that must
survive), active / negated (findings the NER should report).
Usage: python -m scripts.generate_heldout_notes [--n 120]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from medassist.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "samples" / "heldout_notes.json"
SEED = 2026

FIRST = ["Aarav", "Vivaan", "Ishaan", "Kabir", "Rohan", "Siddharth", "Nikhil", "Pranav", "Harish", "Gaurav",
         "Ananya", "Diya", "Kavya", "Meera", "Nisha", "Pooja", "Riya", "Sneha", "Tanvi", "Lakshmi",
         "Farhan", "Imran", "Zoya", "Ayesha", "Gurpreet", "Harleen", "Joseph", "Mary", "Thomas", "Sarah",
         "David", "Emily", "Michael", "Olivia", "Rajesh", "Suresh", "Deepika", "Swati", "Arjun", "Neha"]
LAST = ["Sharma", "Verma", "Gupta", "Iyer", "Nair", "Reddy", "Rao", "Menon", "Pillai", "Kulkarni",
        "Deshmukh", "Joshi", "Chatterjee", "Banerjee", "Mukherjee", "Das", "Bose", "Singh", "Kaur", "Gill",
        "Khan", "Qureshi", "Ansari", "Fernandes", "D'Souza", "Mathew", "George", "Thomas", "Patel", "Shah",
        "Mehta", "Desai", "Agarwal", "Bansal", "Chopra", "Kapoor", "Malhotra", "Saxena", "Tiwari", "Yadav"]
CITIES = ["Mumbai", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Ahmedabad", "Kochi", "Indore", "Nagpur"]
STREETS = ["MG Road", "Park Street", "Anna Salai", "Linking Road", "Residency Road", "Civil Lines", "Banjara Hills"]
# Hospital names are composed so a fresh seed yields names never seen while tuning.
HOSPITAL_PREFIX = ["Sunrise", "Lakeview", "Green Valley", "Riverside", "Silver Oak", "Lotus", "Navjeevan",
                   "Shanti", "Sahyadri", "Kaveri", "Ganga", "Metro", "Lifeline", "St. Joseph's", "St. Mary's"]
HOSPITAL_TYPE = ["Multispeciality Hospital", "Hospital", "Medical Centre", "Clinic", "Nursing Home",
                 "Institute of Medical Sciences", "Super Speciality Hospital"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

# (symptom phrase as written, canonical gold)
SYMPTOMS = [
    ("fever", "fever"), ("high-grade fever", "fever"), ("dry cough", "dry cough"), ("productive cough", "productive cough"),
    ("cough", "cough"), ("breathlessness", "breathlessness"), ("shortness of breath", "shortness of breath"),
    ("chest pain", "chest pain"), ("wheezing", "wheezing"), ("hemoptysis", "hemoptysis"), ("fatigue", "fatigue"),
    ("weight loss", "weight loss"), ("night sweats", "night sweats"), ("headache", "headache"), ("dizziness", "dizziness"),
    ("palpitations", "palpitations"), ("nausea", "nausea"), ("vomiting", "vomiting"), ("abdominal pain", "abdominal pain"),
    ("leg swelling", "leg swelling"), ("orthopnea", "orthopnea"), ("chills", "chills"), ("sore throat", "sore throat"),
    ("confusion", "confusion"), ("syncope", "syncope"),
]
HISTORY = ["hypertension", "type 2 diabetes mellitus", "asthma", "COPD", "tuberculosis", "hypothyroidism",
           "coronary artery disease", "chronic kidney disease"]
NEGATION_TEMPLATES = ["No {s}.", "Denies {s}.", "There is no {s}.", "Patient denies any {s}.", "Negative for {s}."]


def _date(rng: random.Random) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2005)
    return rng.choice([f"{d:02d}/{m:02d}/{y}", f"{d} {MONTHS[m - 1]} {y}", f"{y}-{m:02d}-{d:02d}", f"{d:02d}-{m:02d}-{y}"])


def _phone(rng: random.Random) -> str:
    n = f"{rng.choice('6789')}{rng.randint(100000000, 999999999)}"
    return rng.choice([f"+91 {n[:5]} {n[5:]}", f"+91-{n}", n, f"0{rng.randint(11, 80)}-{rng.randint(2000, 2999)}-{rng.randint(1000, 9999)}"])


def _mrn(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(4)
    if kind == 0:
        v = str(rng.randint(1000000, 9999999))
        return f"MRN: {v}", v
    if kind == 1:
        v = f"{rng.choice(['SMH', 'LMC', 'SJH', 'AC'])}{rng.randint(10000000, 99999999)}"
        return f"UHID {v}", v
    if kind == 2:
        v = f"{rng.randint(2019, 2025)}/{rng.randint(10000, 99999)}"
        return f"Hospital No. {v}", v
    v = str(rng.randint(100000, 999999))
    return f"IP No: {v}", v


def make_note(i: int, rng: random.Random) -> dict:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    doc_first, doc_last = rng.choice(FIRST), rng.choice(LAST)
    mrn_text, mrn_value = _mrn(rng)
    dob, phone = _date(rng), _phone(rng)
    city = rng.choice(CITIES)
    hospital = f"{rng.choice(HOSPITAL_PREFIX)} {rng.choice(HOSPITAL_TYPE)}"
    email = f"{first.lower()}.{last.lower().replace(chr(39), '')}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'yahoo.co.in', 'outlook.com'])}"
    address = f"{rng.randint(1, 250)}, {rng.choice(STREETS)}, {city}"
    age = rng.randint(19, 88)
    sex = rng.choice(["male", "female"])

    name_line = rng.choice([f"Patient: {first} {last}", f"Pt: {last}, {first}", f"Name: {'Mr.' if sex == 'male' else 'Ms.'} {first} {last}"])
    header = [f"{hospital} - {rng.choice(['ED note', 'OPD note', 'Admission note', 'Ward round'])}", f"{name_line}   {mrn_text}   DOB: {dob}"]
    phi = [{"type": "name", "value": f"{first} {last}" if "Pt:" not in name_line else f"{last}, {first}"},
           {"type": "id", "value": mrn_value}, {"type": "date", "value": dob},
           {"type": "name", "value": f"{doc_first} {doc_last}"}, {"type": "facility", "value": hospital}]
    if rng.random() < 0.7:
        header.append(f"Contact: {phone}")
        phi.append({"type": "phone", "value": phone})
    if rng.random() < 0.5:
        header.append(f"Email: {email}")
        phi.append({"type": "email", "value": email})
    if rng.random() < 0.5:
        header.append(f"Address: {address}")
        phi.append({"type": "location", "value": city})

    present = rng.sample(SYMPTOMS, rng.randint(2, 4))
    remaining = [s for s in SYMPTOMS if s[1] not in {p[1] for p in present}]
    negated = rng.sample(remaining, rng.randint(1, 3))
    days = rng.randint(1, 14)
    duration = f"{days} days" if days > 1 else "1 day"
    body = [f"{age}-year-old {sex} presents with {', '.join(p[0] for p in present[:-1])} and {present[-1][0]} for {duration}."]
    for n in negated:
        body.append(rng.choice(NEGATION_TEMPLATES).format(s=n[0]))
    hist = rng.sample(HISTORY, rng.randint(0, 2))
    if hist:
        body.append(f"Known case of {' and '.join(hist)}.")
    spo2, hr, rr = rng.randint(86, 99), rng.randint(62, 128), rng.randint(12, 30)
    sbp, dbp, temp = rng.randint(92, 170), rng.randint(58, 100), round(rng.uniform(36.2, 39.6), 1)
    vitals = f"Vitals: BP {sbp}/{dbp}, HR {hr}, RR {rr}, SpO2 {spo2}%, Temp {temp} C."
    body.append(vitals)
    review = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2024, 2026)}"
    phi.append({"type": "date", "value": review})
    body.append(f"Seen by Dr. {doc_first} {doc_last}. Review on {review}.")

    text = "\n".join(header) + "\n\n" + " ".join(body)
    keep = [p[0] for p in present] + [duration, f"SpO2 {spo2}%"] + hist
    return {
        "id": f"heldout_{i:03d}",
        "text": text,
        "phi": phi,
        "keep": keep,
        "active": [p[1] for p in present],
        "negated": [n[1] for n in negated],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    notes = [make_note(i, rng) for i in range(args.n)]
    out = Path(args.out)
    out.write_text(json.dumps({"_description": __doc__.strip().splitlines()[0], "seed": args.seed, "notes": notes}, indent=1),
                   encoding="utf-8")
    print(f"Wrote {len(notes)} held-out notes (seed {args.seed}) -> {out}")
    print("\nExample:\n" + notes[0]["text"])


if __name__ == "__main__":
    main()

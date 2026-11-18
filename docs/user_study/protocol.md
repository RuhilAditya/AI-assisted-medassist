# MedAssist usability study: protocol

**Goal.** Measure whether people can use MedAssist to understand a case correctly, whether
they find the evidence and explanations clear, and how usable the interface is overall
(System Usability Scale, SUS). This supplies the "user satisfaction" metric the project
brief asks for.

**Ethics.** No real patient data is used: all cases are the built-in demo cases (synthetic
notes on public, de-identified IU X-Ray images). Participation is voluntary and anonymous;
record only a participant code (P01, P02 …) and their background. If your institution needs
approval for student user studies, get it before starting.

## Participants

10–15 people. Aim for a mix and record it in `responses_template.csv`:
- medical students / doctors / nurses (ideally at least 3): `background = clinical`
- engineering students and others: `background = non-clinical`

SUS is stable from about 8–12 participants; report the clinical group separately if n ≥ 3.

## Setup (once)

```bash
python -m scripts.load_demo_cases --analyze --reset
uvicorn api.main:app --port 8000          # open http://127.0.0.1:8000
```

Use a laptop or projector at full screen. Close other GPU-heavy programs (answers take ~30 s).
Reset the demo cases between participants (first command) so earlier questions don't show.

## Session (about 20 minutes per participant)

1. **Intro (2 min).** Read aloud: *"MedAssist helps a doctor review a patient's note and
   chest X-ray. It is a research prototype, not a medical device. We are testing the
   software, not you. Please think aloud."* Do not demonstrate features.
2. **Tasks (12 min).** Give one task at a time; time each from reading to answer. Record
   success (1 = correct, 0.5 = partly, 0 = wrong or gave up) and seconds in the CSV.

   | # | Case | Task | Correct answer |
   |---|---|---|---|
   | T1 | Fever, productive cough, hypoxia | "How urgent is this patient, and what are the two biggest reasons?" | EMERGENCY; NEWS2 7 (SpO₂ 90, RR 24) and/or red flag chest pain + breathlessness |
   | T2 | Same | "Which part of the X-ray made the system think there is a pleural effusion?" | Turns on the effusion heatmap and points at the highlighted region |
   | T3 | Note says no effusion, X-ray disagrees | "Is there anything in this case the doctor should double-check?" | Finds the HIGH conflict: image suggests effusion, note says none |
   | T4 | Same | "Ask the assistant to explain the X-ray. Pick one sentence of the answer and show where its evidence comes from." | Asks a question, hovers/clicks a citation, identifies the source (image or note) |
   | T5 | Pre-employment check | "Does this person need urgent attention?" | No: ROUTINE, no conflicts, findings unlikely |
3. **Questionnaire (5 min).** Participant fills in `questionnaire.md` (paper or form).
4. **Debrief (1 min).** Ask the two open questions; write short notes in the CSV.

## Analysis

Enter every participant as one row in `docs/user_study/responses.csv` (copy the template),
then run:

```bash
python -m scripts.score_user_study
```

It computes SUS (0–100) per participant with mean, SD and 95% CI, the Bangor adjective
rating, task success rates and times, and the explanation-quality items, overall and
by background, and saves `results/phase5_user_study.json`.

**Interpreting SUS** (Bangor et al., 2009): ~68 is average; ≥ 71 "good"; ≥ 80 "excellent".
Report n, mean, SD and CI, and do not over-claim with small samples.

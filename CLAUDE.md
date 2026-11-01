# Palm Oil Ripeness Detection — Project Context

Read this file at the start of every session — it's the persistent memory Claude Code
doesn't otherwise have between sessions. This file stays short on purpose; the plan and
architecture live in their own docs so they don't drift out of sync across three places:

- **`PROJECT_PLAN.md`** — phases, milestones, timeline, risks, decisions log. Read this for
  "what are we doing and why."
- **`SYSTEM_ARCHITECTURE.md`** — components, data flow, evaluation protocol (main split vs.
  leave-one-dataset-out). Read this for "how the system fits together."
- **`data/README.md`** — the dataset merge pipeline specifically: which sources are in,
  taxonomy-mapping decisions per source, how to run the scripts in `src/data/`.
- **`docs/FYP1_Report.pdf`** — the full FYP1 final report as submitted (WIA3002, Sem 2 25/26).
  Primary source for FYP1's literature review, methodology, and baseline results/objectives;
  the summarized numbers in `PROJECT_PLAN.md` §2 trace back to this document.

## What this is
FYP2: Smartphone-Based Palm Oil Ripeness Detection Using Lightweight Vision Models.
Student: Nur Atiqah Maisarah Binti Mohamad Rasi (22003055), Universiti Malaya.
Supervisor: Dr. Muhammad Shahreeza Safiruz Kassim. Co-supervisor: Ahmad Syazwan Ramli (MPOB).

## Non-functional requirements — the acceptance bar for any "final" model
- >= 85% test accuracy
- <= 2s on-device inference
- <= 100MB model size
- Stable across daylight / shadow / low-light
- Supports 224x224 input

(Authoritative source: `PROJECT_PLAN.md` §2.3 — update there first if this ever changes.)

## Conventions
- Python for the ML pipeline (`src/`), Kotlin for the Android app (`app/`).
- Every training run's config + metrics is logged under `results/<run-name>/` — never overwrite a previous run.
- Raw and derived datasets stay out of git (`data/raw/` and `data/processed/` are gitignored) — only commit manifests/scripts, not the images.
- When splitting any new/merged dataset into train/valid/test, split by source photo (or source tree/bunch), never by crop — the FYP1 leakage bug must not repeat.
- Flag explicitly if a change trades off one non-functional requirement against another (e.g. smaller model but lower accuracy).

## Where things stand
See `PROJECT_PLAN.md` **§0 (Current Status)** first — it is the one-screen answer to
"what phase are we in and what is in flight." Then §8 (Decisions Log) for how each
decision was reached, and §9 (Open Questions) for what needs a supervisor. Don't duplicate that here —
update the plan doc directly as decisions are made.

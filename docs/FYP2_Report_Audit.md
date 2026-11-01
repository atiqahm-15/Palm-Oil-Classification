# FYP2 Report Audit — Examiner Review
**Report reviewed:** *Smartphone-Based Palm Oil Ripeness Detection Using Lightweight Vision Models* (Overleaf project "Arxiv & PRIME AI Style Template (1)", `templateArxiv.tex`, pulled live from Overleaf on 2026-09-04)
**Template treated as authoritative (per your instruction):** the Arxiv/PRIME LaTeX template itself (`PRIMEarxiv.sty` + `templateArxiv.tex` conventions) — not a separate faculty document
**Cross-referenced:** FYP1 final report (WIA3002, `docs/FYP1_Report.pdf`), `docs/references.bib`, `docs/report_limitations.md`, `CLAUDE.md`/`PROJECT_PLAN.md` project context

> **Status note, 2026-11-01.** This audit predates Phase 4. **M-1 is superseded**: the Android app was
> built on 2026-09-24 and the report now says so (§1.6, §3.2, §4.6), so RO1 no longer overstates the
> delivered scope. **C-1 is partly applied**: the FastViT entry is corrected to ICCV 2023 in
> `references.bib`; the other four `[UNVERIFIED]` entries are still open. M-2, M-3 and the minor items
> have not been re-checked against the current draft.

---

## 1. Executive Summary

This is an unusually strong FYP report. The methodology section is the standout: leakage is checked and fixed at the (source, photo) level rather than the crop level, duplicate images are MD5-hashed and removed, a split-balancing bug is caught and its cost quantified (+10 points), every headline number is reported as mean ± std over three seeds rather than a single run, and — most tellingly — the report documents its **own** methodological mistake (an under-trained v1 recipe that silently suppressed a fine-tuning-depth ablation, producing a false negative) and shows the corrected result. That kind of self-correction, stated openly rather than buried, is exactly what an examiner wants to see and is rare in undergraduate work.

The writing is precise, appropriately hedged, and free of the inflated claims this rubric is designed to catch. Numbers check out: I independently re-added the dataset table (12,583+4,317+851+416=18,167 ✓), the split table (train+valid+test=18,167 ✓), the LODO gap (94.55−52.11=42.44 ✓), and the depth-attribution percentage ((3.29−1.82)/3.29≈45% ✓) — all correct. I also programmatically checked every LaTeX cross-reference in the document (`\ref`/`\S\ref` against `\label`): all 30 in-text references resolve; nothing will render as a broken "??" in the compiled PDF.

**The one area that is genuinely not submission-ready is the bibliography.** Five of roughly fifteen entries in `references.bib` carry the author's own `[UNVERIFIED]` placeholder notes — including the FastViT citation, which anchors the paper's selected architecture. This is a mechanical fix, not a rewrite, and I've resolved one of them for you below (FastViT is confirmed ICCV 2023).

The second substantive issue is a scope-vs-objective mismatch: RO1 says the project will "develop a smartphone-based... detection system," but Section 3.2 states plainly that the Android app "is designed and specified but not implemented." The report discloses this responsibly throughout (scope, limitations, future work), but the objective statement itself is never reconciled with that scope, which is exactly the kind of gap a viva panel probes first.

Everything else is polish: a few figures/tables are captioned correctly but never explicitly pointed to in the prose ("Table~\ref{...}" missing even though the table sits right there), one short phrase is repeated near-verbatim across two adjacent subsections, and the title's word "Detection" sits oddly against the "classification" language used everywhere else in the body — though I traced this to FYP1's registered project title, so it's likely fixed and not something you can freely change.

**Is it close to submission-ready?** Yes. Fix the bibliography (mechanical), tighten RO1's wording or add one scoping sentence, add a couple of missing in-text figure/table pointers, and confirm with your supervisor that the faculty accepts this arxiv-paper format for the final WIA3002 submission (see §7) — none of these require touching the science, the numbers, or the structure.

---

## 2. Overall Scores

| Category | Score | Why |
|---|---:|---|
| Template Compliance | 88 | Fully consistent with the Arxiv/PRIME conventions themselves (natbib numbered citations, booktabs tables, standard `article` sectioning, no broken refs). Score is capped because I could not confirm from the report alone whether WIA3002/FSKTM also requires the faculty's separate multi-chapter thesis template for final submission — flagged in §7, not treated as a defect. |
| Academic Writing | 93 | Precise, appropriately hedged, no overclaiming. One short phrase repeated near-verbatim across two adjacent subsections (§5.9). |
| Research Structure | 90 | Clean RO→RQ→H→results mapping. RO1's wording doesn't fully match the delivered scope (app not implemented) — see §3, C-1. |
| Methodology | 96 | Genuinely excellent: leakage discipline, MD5 dedup, per-source split balancing with the bug's cost quantified, multi-seed protocol, LODO, documented negative results. Among the strongest methodology sections I'd expect at this level. |
| Results Presentation | 92 | Thorough, internally consistent, well-tabulated. A handful of tables/figures are never explicitly pointed to in the surrounding prose. |
| Discussion | 95 | Interprets rather than repeats; the generalization-gap discussion in particular does real analytical work rather than restating the LODO table. |
| Cross-Chapter Consistency | 93 | Every number I spot-checked across sections matches (see §6). RO1-vs-scope is the one real tension. |
| Figures & Tables | 85 | Numbering/captions are all correct and consistent; five labelled figures/tables are never referenced by number in the text (§5). |
| Citations & References | 68 | Five of ~15 `.bib` entries are self-flagged `[UNVERIFIED]`, including the citation for the selected architecture (FastViT). Must be resolved before submission. |
| **Overall FYP Quality** | **90** | An examiner-ready report held back by a mechanical bibliography cleanup and one objective-wording gap, not by the underlying work. |

---

## 3. 🔴 CRITICAL ISSUES

### C-1 — Five bibliography entries are self-flagged as unverified, including the citation anchoring the selected architecture

**Location:** `docs/references.bib` (synced into the Overleaf project as `references.bib`)

**Problem:** The `.bib` file itself contains explicit `\textbf{[UNVERIFIED]}` notes on five entries:
- `vasu2023fastvit` — "Venue (ICCV vs. CVPR 2023) not confirmed in the source repository"
- `kamal2024mendeley` — "Full author list not confirmed" (author field literally reads `Kamal, and others`)
- `rosbi_outdoor_tenera` — "Full citation details not confirmed"
- `hadeemadadum_kaggle` — "Full citation details not confirmed"
- `harjanto2025yolov8seg` — "Full bibliographic entry not confirmed" (author field reads `Harjanto and others`)

**Why it matters:** FastViT-T8 is the project's selected, headline-reported architecture (94.55% accuracy, the number in the abstract and title claim). An incorrect or unverifiable citation for it is a correctness problem, not a style one — and an examiner who checks even one reference and finds a placeholder note will reasonably wonder about the rest of the bibliography.

**Template/research requirement:** Citations & References — "required bibliographic information," "incorrect citation/reference" is explicitly listed as a 🔴 CRITICAL example in your own rubric.

**Recommended correction:**
- **`vasu2023fastvit` — resolved.** I verified this via web search: FastViT was published at **ICCV 2023** (*Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 5785–5795; also on arXiv as 2303.14189). Suggested corrected entry:
  ```bibtex
  @inproceedings{vasu2023fastvit,
    title     = {{FastViT}: A Fast Hybrid Vision Transformer using Structural Reparameterization},
    author    = {Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    pages     = {5785--5795},
    year      = {2023}
  }
  ```
  Source: [apple/ml-fastvit (GitHub)](https://github.com/apple/ml-fastvit), [ICCV 2023 Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html).
- **`kamal2024mendeley` — partially resolved, needs your confirmation.** I located the dataset page directly: [data.mendeley.com/datasets/424y96m6sw/1](https://data.mendeley.com/datasets/424y96m6sw/1). Its listed title is **"An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality Grading"** — note this differs slightly from the title currently in your `.bib` ("Ordinal Dataset for Ripeness Level Classification **of Oil Palm Fresh Fruit Bunches**"). Please open that page to pull the full author list and correct the title; I did not want to guess the exact author string.
- **`rosbi_outdoor_tenera`, `hadeemadadum_kaggle`, `harjanto2025yolov8seg` — MANUAL VERIFICATION REQUIRED.** I was not able to conclusively locate these (a Harjanto/BINUS YOLOv8n-Seg+ShuffleNetV2 paper did not turn up a clean match in search; the GitHub/Kaggle usernames need their host pages checked directly for full names/dates/DOIs). These need to be resolved from the source pages before submission — do not leave the `[UNVERIFIED]` notes in the submitted `.bib`, since `natbib`/the compiled reference list will otherwise silently present placeholder-quality citations as finished ones.

---

## 4. 🟠 MAJOR ISSUES

### M-1 — RO1's wording promises a "system"; the delivered scope is a model plus an unimplemented app design

**Location:** §1.3 (Research objectives, RO1) vs. §3.2 (App design) and §7.5 (Limitations, L3/scope)

**Problem:** RO1 reads: *"To develop a smartphone-based oil palm Fresh Fruit Bunch (FFB) ripeness detection **system** using lightweight image classification models."* Section 3.2 states directly: *"Status: the application is designed and specified but not implemented within the project period."* What was actually delivered is a trained, exported, benchmarked model (and a documented app **design**) — not a working smartphone system a grader could pick up and use.

**Why it matters:** This is disclosed responsibly and repeatedly elsewhere in the report (Scope §1.4, App design status line, Limitations, Future work item 1), which is good practice — but the objectives statement itself, which is what an examiner reads first and checks the conclusion against, is never adjusted to match. A panel reading RO1 literally can reasonably ask "so was the system built or not?" even though the honest answer is fully available three sections later.

**Template/research requirement:** "Objectives that do not match the actual work" and "whether the conclusion directly answers the research objectives" are both explicit checklist items in your rubric.

**Recommended correction:** Either (a) reword RO1 to scope it to what was built, e.g. *"...to develop and evaluate the core ripeness-classification component of a smartphone-based FFB detection system..."*, or (b) keep RO1 as the originally registered objective (if it can't be changed post-registration) and add one sentence immediately after it in §1.3 stating explicitly that full application implementation is out of scope for FYP2 and reported as future work — pointing forward to §3.2/§7.6. Either fix is a sentence or two, not a rewrite.

### M-2 — Five labelled figures/tables are never explicitly cross-referenced by number in the surrounding prose

**Location:** Figure 1 (`fig:sdlc`, §3.1 pipeline diagram), Table 3 and its companion figure (`tab:archcomparison`, `fig:archcomparison`, §5.2), the lighting-robustness table (`tab:lighting`, §5.6), and the RQ-revisited table (`tab:rq`, §6.1)

**Problem:** I programmatically checked every `\label{...}` against every `\ref{...}`/`\S\ref{...}` in the document. All in-text references resolve correctly (no broken "??"), but these five labels are defined and never pulled into a sentence — the surrounding prose describes the numbers narratively ("FastViT-T8 leads on every accuracy metric...") without ever writing "(Table~\ref{tab:archcomparison})" or "(Figure~\ref{fig:sdlc})".

**Why it matters:** Your own rubric explicitly asks whether every figure/table "is mentioned in the text." Right now a reader has to infer which table is being discussed from proximity alone. It's not wrong — plenty of published papers place a table directly under the heading it belongs to — but it under-uses the numbering the template already gives you for free, and it's the kind of small thing a picky examiner marks down.

**Template/research requirement:** Editorial recommendation (the template doesn't mandate this; it's good practice for a figures/tables checklist).

**Recommended correction:** Add a short explicit pointer to each — e.g., in §3.1: *"Figure~\ref{fig:sdlc} summarizes the end-to-end pipeline described in this section."* In §5.2: *"Table~\ref{tab:archcomparison} reports the full comparison."* Five short insertions, no content change.

### M-3 — Fine-tuning depth is reported as "% of layers," but the recipe itself is specified in "blocks" (K), and the conversion isn't shown in the report

**Location:** §4.6 (training recipe, "specified in blocks rather than layers") vs. §5.3 (depth control, quoting 28.0% / 19.4% / 18.8% of *layers*)

**Problem:** The methodology explicitly and correctly argues that depth should be compared in blocks, not layers, because a fixed layer count means different things per architecture (§4.6, citing Suharjito et al.'s convention). But §5.3 then reports depth as **layer percentages** (28.0%, 19.4%, 18.8%, and later 25.7%, 31.6%, 15.4%) without ever showing, in the report itself, how many total layers each architecture's blocks correspond to, or the block→layer conversion. (I found the underlying per-architecture layer counts referenced only in your internal `report_limitations.md`/`src/training/finetune.py`, not in the report text.)

**Why it matters:** A reader following only the report — not your source repository — cannot verify these percentages or reproduce the depth control from what's written, which is precisely the reproducibility bar Phase C (Methodology) asks about.

**Recommended correction:** Add one small table or footnote in §4.6 or §5.3 giving each architecture's total exposed-block count and total layer count (you already have this data — MobileNetV2/EfficientNetB0: 16 blocks each; FastViT-T8: 10 blocks — the layer totals just need to be stated once).

---

## 5. 🟡 MINOR ISSUES

- **Repeated phrase across adjacent subsections.** "...the underripe/ripe boundary is genuinely ambiguous..." appears almost verbatim in both §3.2 (App design) and §3.3 (Requirements), each pointing back at the other. Consider stating it once (in 3.2) and having 3.3 simply cross-reference it rather than restate it.
- **Title terminology vs. body terminology.** The title and running header use "**Detection**," while the abstract, objectives, and every technical section describe the task as "**classification**" (four-class classification, classify the ripeness). I checked FYP1's report (`docs/FYP1_Report.pdf`) and confirmed "Detection" is the officially registered FYP title carried over from FYP1 — so this likely can't be freely changed. If it can't, one clarifying clause in §1.1 or §1.2 noting that "detection" (the registered project title) refers to the classification task performed would preempt an examiner's terminology question. If it can be changed, "...Ripeness Classification Using..." would match the body exactly.
- **RO2 says "Vision Transformers" (plural); one ViT-family model was ultimately evaluated.** RO2 lists "MobileNet, EfficientNet-Lite, and Vision Transformers" as architecture families to compare; only FastViT-T8 (one hybrid transformer) was evaluated, for the well-documented reasons in §4.4.2. Not a real problem — the substitution is transparently explained — but the plural in the objective slightly overstates what RO2 as literally worded promised. No action needed beyond awareness; not worth editing given how thoroughly §4.4.2 already explains it.
- **A precision note that's already handled correctly, flagged for your own visibility only:** `report_limitations.md` notes that GPU nondeterminism caused calibration deltas to vary between reruns (+1.41 vs. +1.52 for EfficientNetB0) and recommends quoting to at most one decimal place. I checked — the final report's calibration table (Table, §5.9) already reports to one decimal throughout. No fix needed; this is confirmation that an internal caution was correctly carried through to the final draft.

---

## 6. Cross-Chapter Consistency Report

| Item | Location A | Location B | Consistent? | Required Action |
|---|---|---|---|---|
| Headline accuracy (94.55% ± 0.31%) | Abstract | §5.1 Headline result, Table (NFR, §3.3), Table (RQ, §6.1), Conclusion | ✅ | None |
| LODO accuracy (52.11%) / gap (42.44 pts) | Abstract | §5.4 (LODO table + discussion), §6.1 RQ table, §6.2 discussion | ✅ Arithmetic verified: 94.55 − 52.11 = 42.44 | None |
| Dataset total (18,167 images) | Abstract | §4.5.1 Sources table (12,583+4,317+851+416=18,167, verified), §4.5.2 split table (12,716+2,729+2,722=18,167, verified) | ✅ | None |
| Unripe recall repair (0.38 → 0.940) | Abstract | §5.1 Headline result | ✅ (internally consistent; FYP1's 0.38 figure could not be independently located in FYP1's extracted report text — it may be embedded as an image/screenshot there rather than as text) | ❓ Manually confirm the 0.38 figure against FYP1's own results table/screenshot before final submission |
| Deployed model size (6.1 MB, float16) | Abstract | §3.3 NFR table, §5.5 TFLite table, Conclusion | ✅ | None |
| Depth-attribution ("~45%", 3.29→1.82 pts) | Abstract | §5.3 (verified: (3.29−1.82)/3.29 ≈ 44.7% ≈ 45%) | ✅ | None |
| Research objectives (RO1–RO3) | §1.3 | Verified word-for-word identical to FYP1's §1.3 objectives | ✅ Matches your stated intent to keep FYP1/FYP2 consistent | None |
| RO1 ("develop a system") | §1.3 | §3.2 status line ("designed and specified but not implemented"), §7.5 L3 | ⚠️ Partially | See M-1 — reconcile wording or add scoping sentence |
| Fine-tuning depth (K, blocks) vs. reported % | §4.6 (blocks) | §5.3 (layer %) | ⚠️ Not shown | See M-3 — add block→layer conversion table |
| Duplicate-row defect | §4.5.2 (test split: 2,722 rows / 2,695 unique) | §5.8 (impact table, deltas of −0.03 to −0.08 pts) | ✅ Consistent, and correctly scoped as reporting error not training error | None |
| FastViT citation | §2.3 (Lightweight architectures) | `references.bib` | ❌ Bib entry self-flagged `[UNVERIFIED]` | See C-1 |

---

## 7. Template Compliance Report

| Requirement | Template Requirement | Your Report | Status | Action |
|---|---|---|---|---|
| Document class / package | `article` + `PRIMEarxiv.sty` | Used exactly as specified | ✅ Compliant | None |
| Citation style | `natbib`, numeric, sort&compress | `\citep`/`\citet` used consistently throughout | ✅ Compliant | None |
| Tables | `booktabs` (toprule/midrule/bottomrule) | Used consistently in every table | ✅ Compliant | None |
| Figure/table numbering & cross-refs | Automatic via `\label`/`\ref` | 40 labels, 30 refs, zero broken references (verified programmatically) | ✅ Compliant | None |
| Figures/tables mentioned in text | Not template-mandated; good practice | 5 of ~15 figures/tables never explicitly pointed to by number | ⚠️ Partially compliant | See M-2 |
| Section structure (Abstract→Intro→Related Work→Methods→Results→Discussion→Conclusion→Refs→Appendix) | Standard for this template family | Followed exactly | ✅ Compliant | None |
| Header/title block | `fancyhdr`, running head, title/author/affiliation block | Present and correctly filled | ✅ Compliant | None |
| Margins, font, exact spacing | Set by `PRIMEarxiv.sty` | Not independently re-measured — these are enforced automatically by the class at compile time, not something to hand-check | ✅ Compliant by construction | None |
| **Faculty-level FYP submission format** (chapters, declaration page, TOC, front/back matter) — **not part of this template** | Universiti Malaya / FSKTM WIA3002 conventionally has its own separate multi-chapter thesis template for some submission stages | This report is a single-document, paper-style write-up (no chapters, declaration, or TOC) | ❓ Manual verification required | **Confirm with your supervisor/coordinator whether the arxiv-paper format is accepted for your actual WIA3002 final submission, or whether a separate chaptered document is also required.** I'm flagging this once, neutrally — you told me to treat this template as authoritative, so I have not penalized the report for lacking chapter structure, but this is worth a two-line email to your supervisor before you consider the format question closed. |

---

## 8. Chapter-by-Chapter Review

*(Mapped onto your report's actual sections, since it's structured as a paper rather than numbered chapters: §1 Introduction, §2 Literature Review, §3–4 Methodology, §5 Results and Analysis, §6 Discussion and Conclusion, References, Appendix.)*

### Introduction (§1)
**Strengths:** Clean Context→Problem→Gap→Objectives flow; RQs and hypotheses (H1–H4) are stated explicitly and each is later resolved by name in the discussion — a level of explicit hypothesis tracking that's unusually rigorous for this level.
**Problems:** RO1 wording vs. delivered scope (M-1). Title/body terminology ("Detection" vs. "classification," minor, likely fixed by FYP1's registered title).
**Required changes:** Address M-1.
**Optional improvements:** One clarifying clause on "detection" vs. "classification" terminology.

### Literature Review (§2)
**Strengths:** Genuinely synthesizes rather than lists — the three-pattern synthesis after the comparison table (single-source/single-run reporting, ripe/overripe as the recurring failure mode, no study combining a size/latency budget with a mobile ViT) is exactly the "compare and identify gaps" behavior your rubric asks for, not just paper-by-paper description. The research gap statement is explicit and ties directly to the project's three contributions.
**Problems:** None substantive found.
**Required changes:** None.
**Optional improvements:** None needed.

### Methodology (§3–4)
**Strengths:** This is the strongest chapter. Explicit leakage discipline (split by source photo, not crop), MD5 deduplication both within and across sources, a caught-and-quantified split-balancing bug, architecture substitutions documented with the specific technical failures that forced them (not hand-waved), a shared recipe held identical by construction, and a full reproducibility register (config files, versioned artifacts, refusal to pool across recipe versions).
**Problems:** M-3 (block→layer conversion not shown in-report).
**Required changes:** Address M-3.
**Optional improvements:** None needed; this section does not need simplification.

### Results and Analysis (§5)
**Strengths:** Every headline number is mean±std over 3 seeds. The training-schedule correction (§5.3) and the "falsified control" discussion (depth ablation run under the broken v1 recipe) are the report's best material — a real methodological contribution, correctly framed as a finding rather than buried as an error. The duplicate-row-defect section is a model of exactly the kind of measurement-integrity check most undergrad reports skip entirely.
**Problems:** M-2 (missing in-text figure/table pointers).
**Required changes:** Address M-2.
**Optional improvements:** None.

### Discussion and Conclusion (§6)
**Strengths:** The generalization-gap discussion interprets the LODO result (convention-fitting vs. capacity limitation) rather than restating the table, and explicitly follows through the counterintuitive implication that under-training was accidentally acting as regularization. Limitations (L1–L8) are specific and numbered, not generic boilerplate. Future work items are concrete and each tied to a named limitation.
**Problems:** None substantive found beyond what's already logged as M-1 (which surfaces again here since the conclusion correctly reports the app as undelivered — consistent with the results, just not with RO1's original wording).
**Required changes:** None beyond M-1.
**Optional improvements:** None.

### References
**Strengths:** Correct `natbib` numeric formatting throughout; citation density is appropriate (not padded, not sparse).
**Problems:** C-1 (five self-flagged unverified entries).
**Required changes:** Resolve all five entries before submission.
**Optional improvements:** None until C-1 is resolved.

### Appendices (Figure index, Run index)
**Strengths:** The run index mapping every headline result to a specific run directory/artifact is an excellent reproducibility touch — genuinely goes beyond what most FYP reports include.
**Problems:** None found.
**Required changes:** None.
**Optional improvements:** None.

---

## 9. Sentence-Level Wording Corrections

The prose is already strong throughout, so this list is short — most sentences don't need touching.

**Original (§1.3, RO1):**
> "RO1 -- To develop a smartphone-based oil palm Fresh Fruit Bunch (FFB) ripeness detection system using lightweight image classification models"

**Suggested:**
> "RO1 -- To develop and evaluate the core ripeness-classification component of a smartphone-based oil palm Fresh Fruit Bunch (FFB) detection system, using lightweight image classification models"

**Reason:** Scopes the objective to what was actually delivered (see M-1), without changing its substance.

---

**Original (§3.3, Requirements bullet):**
> "return the predicted class together with a per-class confidence breakdown rather than a single label, since the underripe/ripe boundary is genuinely ambiguous (Figure~\ref{fig:appmockup}, Section~\ref{sec:appdesign});"

**Suggested:**
> "return the predicted class together with a per-class confidence breakdown rather than a single label, for the reason given in Section~\ref{sec:appdesign};"

**Reason:** Removes the near-duplicate restatement of the ambiguity rationale that already appears in full in §3.2, one paragraph earlier.

---

## Final Priority List

### MUST FIX BEFORE SUBMISSION
1. Resolve all five `[UNVERIFIED]` bibliography entries in `references.bib` — start with FastViT (fix supplied above, C-1) and the Mendeley dataset title/authors (page located, C-1); manually verify the remaining three.
2. Reconcile RO1's wording with the actual delivered scope — either reword RO1 or add one explicit scoping sentence in §1.3 pointing to §3.2/§7.6 (M-1).
3. Confirm with your supervisor whether the WIA3002/FSKTM final submission accepts this arxiv-paper single-document format, or requires the faculty's separate chaptered thesis template (§7).

### SHOULD FIX
1. Add explicit "(Table~\ref{...})"/"(Figure~\ref{...})" pointers for the five figures/tables that are currently captioned but never referenced by number in the prose (M-2).
2. Add a short block→layer conversion table/footnote so the fine-tuning-depth percentages in §5.3 are reproducible from the report text alone (M-3).
3. Manually confirm the FYP1 unripe-recall baseline (0.38) against FYP1's own results table before quoting it again in the abstract/headline result.

### OPTIONAL POLISH
1. Trim the repeated "underripe/ripe boundary is genuinely ambiguous" phrasing between §3.2 and §3.3.
2. Add one clarifying clause reconciling the title's "Detection" with the body's consistent "classification" terminology (only if the registered project title can't itself be changed).
3. Consider a brief note that RO2's "Vision Transformers" (plural) is served by one hybrid transformer, for the documented reasons in §4.4.2 — no action needed, just something to be ready to explain if asked in a viva.

---

## Submission Readiness: 88/100

What's holding it back from higher: the bibliography (C-1) is the single largest deduction — five self-flagged placeholder citations is not a stylistic nitpick, it's incomplete work that the author's own notes already flag as outstanding, and it touches the citation for the project's selected architecture. The RO1-vs-scope gap (M-1) is the second deduction, because it's the kind of mismatch a viva panel finds in the first five minutes. Everything else — the missing figure/table pointers, the depth-percentage traceability gap, the repeated phrase — is real but minor polish that doesn't touch the science, the numbers, or the argument, all of which I found to be careful, consistent, and unusually honest about the project's own limitations. Fix the bibliography and tighten RO1, and this reads as a complete, submission-ready report.

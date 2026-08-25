---
name: h3-character-eval
description: Measure whether generated videos preserved a character's declared visual traits — the contract Ref2VA retention_analysis asserts. Use when comparing styles, prompts, models, or skill versions on character consistency; when validating a character card before production use; or when a subjective "the character drifted" needs a number. Builds a trait contract from the character card, renders via reference-to-video, audits frames with a vision model under three calibration controls (positive control, two-pass stability, hard negatives), and reports pass rates in the retention-marker vocabulary (fully_preserved / partially_preserved / weak_reference).
compatibility: Portable to any agent that can call the MiniMax video generation API (or MiniMax Hub video tools), call a vision-capable chat model (e.g. MiniMax-M3), and extract video frames (ffmpeg). No other external services.
---

# H3 Character Retention Eval

Every Ref2VA prompt promises per-subject retention (`fully_preserved`, ...) and
the style skills name the failure modes — identity swap, face fusion, trait
drift — but nothing in the toolchain measures whether a delivered video honored
the promise. This skill is that measurement. Its design was validated end to
end on an 8-character × 3-style × 2-model grid (144 renders): the calibration
controls below caught unrendered promises, vision-judge misses, and
badly-authored traits before any of them could pollute a score, reaching judge
stability 0.992 and specificity 1.000. Method details, the judge prompt, and
the pitfalls live in [`references/eval-protocol.md`](references/eval-protocol.md).

Core idea: **score the model against its own contract.** Traits are declared
from the character card, the render is asked to preserve them, and a vision
judge checks each trait in sampled frames — with controls that measure the
judge itself, because an uncalibrated judge produces confident nonsense.

## STEP 1 — Author the trait contract

For each character, write 4–6 traits from its reference card. Every trait must
be:

- **Binary and visually checkable in a single frame** ("red collar", "black
  patch surrounding one eye") — never a vibe ("cute", "consistent").
- **Style-invariant identity content** — markings, colors, accessories,
  asymmetries. Never rendering style ("oversized cartoon eyes" *correctly*
  disappears in a photoreal restyle and would score as a false failure).
- **Distinctive against a confusable character** — "white chest patch" is true
  of half of all dogs and will leak in the negative control; compound the
  feature with color/species context until it discriminates.

Multi-character scenes get **binding traits** instead of appearance traits:
"the red collar is worn by the dog, not the cat", "exactly two animals",
"clearly different body shapes, not blended" — swap and fusion only exist with
two or more subjects.

## STEP 2 — Positive control: audit the card itself

Before rendering anything, run the vision judge on the character card against
its own trait list (protocol in the reference file). A trait the card does not
clearly show is **not checkable** — the image model that made the card may not
have rendered it, or the wording may be falsified by the card (a "solid black
coat" with a white chest patch). Drop non-present traits from all scoring.
Scoring a video against a promise the card never made is noise.

## STEP 3 — Render the grid

Render each character through reference-to-video (`role: reference_image`,
the card(s) as references) with a **fixed scenario** across every arm so the
variable under test (style, prompt, model, skill version) is the only thing
that changes. Always include the identity-anchor clause: references are for
identity mapping only — preserve every listed marking, color, and accessory;
restyle the rendering, never the anchors. Use **n≥3 renders per cell** for any
claim beyond screening; a single render confounds render variance with a real
effect.

## STEP 4 — Two-pass audit with tie-breaks

For each video, sample 5 frames at spread timestamps and ask the vision judge
for a strict-JSON verdict per trait: `present` / `absent` / `unsure` (exact
prompt in the reference file; temperature 0). Then judge the **same video
again on a disjoint frame sample**. Where the two passes disagree on any
trait, judge a third disjoint sample and take the per-trait majority; no
majority means `unsure`. `unsure` counts as a failure but is tallied
separately.

## STEP 5 — Calibrate the judge, always

Report these next to every headline number — they are what make it readable:

- **Stability** = agreement rate between pass A and pass B over checkable
  traits. Any gap between arms smaller than (1 − stability) is judge noise,
  not a finding. In validation this control caught the run's only two
  apparent failures as judge misses.
- **Specificity** = 1 − leak rate when each video is judged against a
  *different, confusable* character's traits (same species where possible;
  never a character that appears in the video). Foreign traits should come
  back `absent`; leaks mean the traits are too generic to trust.

## STEP 6 — Scorecard in the retention vocabulary

A cell's score is its mean trait-pass rate over checkable traits and reps.
Map it onto the markers the Ref2VA contract uses: ≥0.9 `fully_preserved`,
≥0.5 `partially_preserved`, else `weak_reference` — and present it as
**promised vs delivered** (the promise is `fully_preserved` everywhere). For
arm comparisons, use a sign test over the paired per-cell differences rather
than eyeballing means. List every failed trait by cell: "which trait, which
style" is the actionable output, not the aggregate.

# Eval protocol — judge prompt, controls, and pitfalls

This is the measurement layer behind `SKILL.md`, with the exact judge prompt
and the arithmetic. Everything here was validated on a real grid: 8 characters
(dog / cat / bird / exotic; photo / cartoon / 3D-render source cards; one
two-subject binding row) × 3 target styles (photoreal, Pixar-style 3D,
hand-drawn) × 2 video models × 3 replicates — 144 renders, 316 judge calls.

## The judge prompt

Send the sampled frames plus this text to a vision-capable chat model at
temperature 0 (validation used MiniMax-M3; strip any thinking block before
parsing):

```
You are auditing whether a generated video preserved a character's declared
visual traits. The attached images are frames sampled from one video. For each
numbered trait, answer:
- "present": the trait is clearly visible in at least one frame
- "absent": the subject is visible but the trait is missing, changed, or on
the wrong individual
- "unsure": the frames do not let you tell

The video may be stylized (cartoon, 3D, hand-drawn): judge the trait's identity
content (marking, color, accessory, asymmetry), not its rendering style.

Traits:
{numbered trait list}

Reply with ONLY a JSON object mapping each trait number to its verdict, e.g.
{"1": "present", "2": "absent"}. No other text.
```

Frame samples (fractions of the clip's duration):

- pass A: 0.02, 0.25, 0.50, 0.75, 0.97
- pass B: 0.12, 0.37, 0.62, 0.87, 0.90
- tie-break pass C (only where A and B disagree): 0.06, 0.30, 0.55, 0.80, 0.94

Per-trait verdict: A if A==B, else the majority of {A, B, C}; no majority →
`unsure`. `unsure` fails the trait but is tallied separately from `absent`.

## The three controls, and why each exists

**1. Positive control (card audit) — defines the checkable set.** Judge each
card against its own traits before any video is scored; drop traits not
`present`. Two distinct causes show up in practice: the card's image model
did not render a requested detail (asymmetric ears, a leg band, a paw
marking), and trait wording the card itself falsifies (a "solid black coat"
on a card with a white chest patch). In validation this pruned 6/42 traits on
the first pass and 1/35 after traits were re-authored against the cards.

**2. Two-pass stability — measures the judge.** Stability = per-trait
agreement between passes A and B across all videos. Report it with every
result and refuse to read any arm-vs-arm gap smaller than (1 − stability). In
validation the run's only two apparent retention failures sat exactly on A/B
disagreements — human frame inspection confirmed the traits were present, so
the "failures" were judge misses the control had already flagged. Stability
was 0.969 with loosely-authored traits and 0.992 after the re-authoring.

**3. Hard-negative specificity — measures the traits.** Judge each video once
against a *different* character's trait list and expect `absent` everywhere.
Two rules learned the hard way: pick a **confusable** foreign character (same
species where possible — corgi↔pug, black cat↔ginger tabby), and **never one
whose subject appears in the video** (a two-character video judged against
one of its own members "leaks" its genuine traits and reads as a false
failure of the control). Leaked traits are too generic — rewrite them with
color/species context until they discriminate. Validation: 0.769 specificity
with generic traits and an overlapping pairing; 1.000 after both fixes.

## Scoring

- Cell = one character × one arm (style/model/prompt variant). Cell score =
  mean over replicates of (traits `present` ÷ checkable traits), pass-A/B/C
  verdicts as above.
- Markers: ≥0.9 `fully_preserved`, ≥0.5 `partially_preserved`, else
  `weak_reference`. The promise is `fully_preserved` everywhere — report
  promised vs delivered.
- Arm comparison: exact two-sided sign test over the paired per-cell
  differences (ties dropped). Means without the sign test overstate small
  grids.
- Always list failed traits by cell. "The crest drops in stylized restyles of
  birds" is actionable; "0.97 overall" is not.

## Pitfalls checklist

- A trait the card doesn't show scores the card generator, not the video
  model — that's what the positive control removes.
- Style traits ("oversized cartoon eyes") fail *correctly* under restyling
  and poison the score — identity content only.
- n=1 per cell confounds render variance with real effects: in validation,
  every sub-1.0 cell at n=3 had two perfect sibling replicates.
- Sub-second single-frame checks miss traits that flicker; "present in at
  least one frame" over 5 spread frames is deliberate.
- If the judge and the generator share a vendor, the controls are the
  mitigation, not brand independence — say so when reporting.

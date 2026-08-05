# Text and Pencil Storyboard Guidelines

## STEP 6: Text Storyboards Document (Default) + Pencil Image Storyboards (Opt-in)

After the Step 5.5 self-check passes, show a storyboard-mode choice card before producing any storyboard artifact:

- **Text storyboards document only (default, recommended)** — one canvas text node containing all shot storyboards as in-document sections. Mirrors the half-narrated-drama storyboard structure: per-shot fields (title / hook / scene / characters / spatial anchors / continuity / performance) plus Pixar's per-panel four-quadrant content + optional ASCII layout. Carries the full quality-control payload at near-zero cost. The video model selected in Step 7 reads this directly as the per-shot rendering reference.
- **Text storyboards document + multi-panel pencil image (visualization mode, opt-in)** — the text storyboards document is still produced as the authoritative artifact, AND one multi-panel pencil image is generated per shot for human review. Higher cost, useful when the user wants a visual preview before committing to video generation, or when squash-and-stretch / pose silhouette is the main risk and the user wants to pre-check it visually.

Store the chosen storyboard mode in the Project Brief and reuse it in Step 7, Step 9, and the Regeneration discipline.

### Default path: single text storyboards document

Generate one canvas text node named `<title> text storyboards` (one document for the whole short). This document is the authoritative rendering reference for Step 7 even when pencil images are also produced. The structure mirrors the half-narrated-drama storyboard — every shot is a section in the same document, so the user can read cross-shot continuity without node-hopping.

Document top matter (header block at the top of the document):

- Project title, approved video model, approved resolution, storyboard mode, and self-check status (e.g. `shot-table self-check: passed at <timestamp>`).
- A short table-of-contents listing every shot, its hook, and its section anchor (`S01`, `S02`, …) so the user can jump.

Per-shot section structure (one `##` heading per shot, in shot order). Every section is mandatory to contain these fields, in this order — direct adaptation of the half-narrated-drama storyboard:

1. **Shot title & duration** — short human-readable title for the shot, plus `S<N> / <duration>s` (e.g. `S03 / 6s`).
2. **Hook type** — one of the controlled vocabulary: `setup` / `visual-joke` / `reversal` / `reveal` / `callback` / `suspense` / `tender` / `chase` / `expression-beat` / `climax`. Used by the per-episode hook distribution self-check.
3. **Scene & characters** — exact scene card name and exact on-screen character names (binding to character cards).
4. **Spatial anchor card** (mandatory, four sub-fields — directly adapted from half-narrated):
   - `Fixed landmarks` — named landmarks and their screen-relative positions (e.g. `door-frame: right third`, `kitchen-island: center bottom`).
   - `Character positions (camera view)` — for every on-screen character, screen-relative position, facing direction, and initial pose.
   - `Exited character status` — characters who were on screen in the previous shot but not in this one, with their off-screen position and reason.
   - `Lighting baseline` — inherited key/fill/rim direction from the scene card, plus per-shot modifier.
5. **Continuity** (mirrors half-narrated's handoff fields):
   - `Continuity from S(N-1)` — one or two sentences referencing the previous shot’s ending state.
   - `Continuity to S(N+1)` — one sentence setting up the next shot’s opening.
6. **Double-binding** — `[char:角色名-01] [char:角色名-02] ... [scene:场景名] [hook: visual-joke]` — exact character card names, scene card name, hook type. These are storyboard-only reference markers; the video model strips them at render time.
7. **Per-panel four-quadrant content** (one block per panel, in time order; this is the Pixar per-second directive, kept verbatim from the table row):
   - `Timecode` — e.g. `0–1s`.
   - `Pose + Expression` — concrete body posture, silhouette, key prop grip, eye-line, facial expression path; for elastic beats, explicitly call out squash / stretch / anticipation / overshoot. This is the largest section per panel and is what the video model reads as the visual beat.
   - `Camera` — shot size, camera movement (push / pull / pan / tilt / handheld-shake / locked / orbit), Dutch angle note when applicable.
   - `Audio + Anchor` — audio cue (`♪ narration: ...` / `dialogue: ...` / `SFX: ...` / `silent`) and spatial anchor note (`door-frame: right third` / `Mia: center midground facing camera`).
   - Performance notes (mirrors half-narrated): for narration seconds, mark `narrator-mouth-closed: true`; for on-screen dialogue, mark `mouth-open: speaker` and describe expression path / eye-line / body-action changes.
8. **Layout rules** (apply per shot):
   - 3-second shot → 3 panels (1 per second).
   - 4-second shot → 4 panels.
   - 5-second shot → 5 panels.
   - 6-second shot → 6 panels.
   - 7+ second shot → one panel per second; for sub-second critical beats, add an extra mini-panel such as `2.0–2.5s` only when the beat is the hook of the shot.
   - Panels must cover the full shot duration from first frame to last frame with no time gaps.
9. **Per-panel binding**:
   - Bind the exact character cards listed in the table row to lock appearance, face, hairstyle, body proportions, costume, signature props, and role identity. Use the same character names as the table.
   - Bind the exact scene card listed in the row to preserve environment, props, landmarks, movement paths, and spatial logic.
10. **Optional ASCII layout block (highly recommended, free)**:
    - Append a small ASCII sketch per panel (or one combined sketch for the whole shot) so the user can scan the spatial layout in seconds without rendering an image. Example:
      ```
      [0-1s]  Mia (L, mid)         door-frame (R)
              ──kneels, hands on apple basket──
              cam: low push-in, locked
              audio: silent | anchor: basket center-bottom
      [1-2s]  ...
      ```
    - The ASCII block is informational only; the video model reads the structured `Per-panel four-quadrant content` above, not the ASCII.
11. **Storyboard-only markers**:
    - When a beat is critical, append `[BEAT]` after the panel timecode.
    - When a panel must handoff a specific state to the next panel or the next shot, append `[HANDOFF → ...]` with a short label such as `[HANDOFF → S04 opening]`.

Per-shot section template (copy-paste skeleton, valid for any shot):

```markdown
## S03 / 6s — Title: 奶奶把苹果筐递给 Mia

- **Hook type**: reveal
- **Scene & characters**: scene:kitchen | char:Mia, char:Grandma
- **Spatial anchor card**:
  - Fixed landmarks: door-frame (right third), kitchen-island (center bottom)
  - Character positions: Mia (L, midground, facing camera) | Grandma (R, foreground, facing Mia)
  - Exited character status: —
  - Lighting baseline: warm overhead key + cool bounce right
- **Continuity from S02**: 奶奶弯下腰从中岛拿起苹果筐
- **Continuity to S04**: Mia 接住筐转身，门铃响起
- **Double-binding**: [char:Mia] [char:Grandma] [scene:kitchen] [hook:reveal]

### Per-panel four-quadrant content

#### 0–1s
- Pose + Expression: 奶奶弯腰双手持筐；Mia 左侧站姿，眼神好奇
- Camera: locked medium shot, eye-level
- Audio + Anchor: silent | Mia: L midground | basket: center bottom
- Performance: [BEAT]

#### 1–2s
- Pose + Expression: 奶奶手臂伸向 Mia，筐倾斜；Mia 双手前伸准备接
- Camera: locked medium shot, eye-level
- Audio + Anchor: ♪ SFX: basket rustle | anchor: door-frame: right third
- Performance: [HANDOFF → S04 opening]

#### 2–3s
...

### ASCII layout (optional)
[0-1s]  Grandma (R, fg)        door-frame (R, bg)
        ──lifts basket──        Mia (L, mid)
        cam: locked | silent
[1-2s]  ...
```

After all sections are written, place the document on canvas and move directly to Step 7. Do not call any image generation model in default mode.

### Shot-level extraction (heavy-iteration mode)

The default single-document form is optimized for reading and cross-shot continuity. When the user flags a specific shot for heavy iteration (typically climax / chase / slapstick beats where the per-panel content needs many rounds of revision), extract that section into a standalone text node so iteration is localized:

- User signal: at any time after Step 6, the user says things like "let me focus on S05", "S05 needs rework", "extract S05", or selects a shot during the storyboard approval choice card.
- Extraction mechanics:
  1. Create a new canvas text node named `<title> S05 text storyboard (extracted)`.
  2. Move the full content of the `## S05` section from the document into the new node.
  3. In the document, replace the `## S05` section with a one-line placeholder: `> S05 — extracted to standalone node (see `<title> S05 text storyboard (extracted)`)`.
  4. Step 7 reads from the extracted node for S05; all other shots still read from the document.
- Re-integration: when the user is satisfied, the standalone node is folded back into the document (replace the placeholder with the latest content) and the standalone node is archived.
- Multiple extracted shots: each shot gets its own standalone node; the document tracks them with placeholders.

The extraction mechanism exists because independent nodes are best used by need, not by default — but they remain available whenever iteration pressure is high on a specific shot.

### Opt-in path: multi-panel pencil image storyboards (visualization mode)

If the user picked the visualization mode in the storyboard-mode choice card, ALSO produce one multi-panel pencil storyboard image per table row on top of the text storyboards document. The text storyboards document remains the authoritative rendering reference; the pencil images are human-review-only.

For each pencil image storyboard:

- **Double-binding labels (top-right corner, mandatory on image)**:
  - `[char:角色名-01] [char:角色名-02] ...` — exact character card names used in this row.
  - `[scene:场景名]` — exact scene card name.
  - `[shot: S03] [dur: 6s] [hook: visual-joke]` — shot ID, duration, and hook type.
  - These labels are storyboard-only reference markers; they are stripped at video render time.
- Bind the exact character cards listed in that row to lock character appearance, face, hairstyle, body proportions, costume, signature props, and role identity.
- Bind the exact scene card listed in that row to preserve environment, props, landmarks, movement paths, and spatial logic.
- Convert every per-second directive in the row into one storyboard panel or one clearly labeled beat panel; for a 4-second shot, normally create 4 panels; for a 6-second shot, normally create 6 panels; for sub-second critical beats, add extra mini-panels only when needed.
- **Panel physical layout (mandatory)**:
  - 3-second shot → 1×3 strip.
  - 4-second shot → 2×2 grid.
  - 5-second shot → top row 3 + bottom row 2.
  - 6-second shot → 2×3 grid.
  - 7+ second shot → 3 rows, balanced panels.
  - Each panel occupies the same canvas area; do not let one panel dominate.
- **Per-panel four-quadrant content (mandatory)**:
  - Top-left: timecode (e.g. `0–1s`).
  - Top-right: pose + expression sketch (the largest area; the actual visual beat).
  - Bottom-left: camera icon + movement arrow (push/pull/pan/orbit/locked) and a tiny note for Dutch angle.
  - Bottom-right: audio cue (e.g. `♪ narration: "I knew it."` / `SFX: door creak` / `silent`) and anchor note (e.g. `door-frame: right third`).
- Arrange panels in reading order inside the same single-shot storyboard image; do not merge multiple different shots into one image.
- Each panel must mark its timecode, such as `0–1s`, `1–2s`, and show the corresponding pose, expression, action, camera movement, prop position, SFX cue, and continuity handoff.
- Output pure black-and-white pencil line-art only: no color, no final-render lighting, no polished 3D render.
- Mark the storyboard image with the shot number and include camera-movement icon / marker per panel when useful.
- Include storyboard-only marks when useful: pencil construction lines, action arrows, camera-path icon, timing marks, and small notes.
- Keep the draft as a video-render reference asset only, not final art.

### Storyboard approval (both modes)

After all text storyboard sections (and pencil images, if visualization mode is on) are produced, place them on canvas in shot order, group them as:
- `<title> text storyboards` (default mode, single document), OR
- `<title> text storyboards + multi-panel pencil storyboards` (visualization mode, group the text document and the pencil images separately because pencil images contain double-binding labels, ASCII labels, and shot numbers that the text document does not).

Show a user choice card:

- Approve storyboards and render shot videos (recommended)
- Extract / re-integrate a shot (move section between document and standalone node)
- Redraw selected pencil storyboard (visualization mode)
- Fix character consistency
- Fix scene logic
- Fix camera marker
- Fix audio/anchor markers

### Storyboard generation failure fallback (visualization mode only)

If a pencil image storyboard cannot be produced at the required quality (e.g. layout collapses, labels illegible, panels merged, character inconsistency), apply the following escalation before asking the user:

1. **First retry**: regenerate the same shot storyboard with a tightened prompt that explicitly mentions the four-quadrant layout, the `[char:…] [scene:…] [shot:…]` labels, and the per-panel content rules.
2. **Second retry**: drop the bottom-right audio/anchor quadrant text (keep it as a blank cell with a tiny `♪` mark) to reduce text load; this usually fixes illegible labels without losing the visual beat.
3. **Third retry**: reduce panel count by one (e.g. 6 panels → 5 panels by merging the two least-actionable seconds) and simplify camera icons to single arrows.
4. **After three failed attempts on the same shot**: pause and ask the user with a choice card:
   - Switch to a block-color storyboard (gray boxes for poses, no pencil lines) for the failing shot only.
   - Drop the pencil image for the failing shot and rely on the text storyboards document alone for that row.
   - Split the failing shot into two shorter shots in Step 5 and re-run Step 5.5.
   - Manually supply a reference image to bind instead of generating.

In default text mode this whole fallback is unnecessary — text storyboards fail only when the model cannot produce coherent structured text, in which case return to Step 5 to revise the table row.


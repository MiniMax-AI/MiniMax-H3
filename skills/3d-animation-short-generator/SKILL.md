---
name: 3d-animation-short-generator
description: |
  Create complete stylized 3D animated shorts from a story idea through an ordered production workflow covering project brief, story outline, character and environment cards, standardized shot planning, text or optional pencil storyboards, video-model selection, single-shot generation, assembly, BGM matching, and final review. Use when the user wants an end-to-end narrative animation workflow with strong character consistency, scene continuity, timing, camera, performance, and audio control. Not for single images, simple edits, photorealistic live action, or one standalone clip.
---

# 3D Animation Short Generator

Use this Skill when the user wants a complete story-first animated short workflow, from one-line idea to final edited video. The workflow must place every major artifact on the canvas in production order and pause at creative gates with user choice cards before expensive or high-impact steps.

Core rule: **story first, ask screen size and total duration with choice cards immediately after user intake, ordered canvas artifacts, every required confirmation via choice card, fixed order after character/scene cards: six-column standardized shot table with per-second directives + audio cues + spatial anchor chain → shot-table self-check gate → one single text storyboards document with one section per shot (multi-panel pencil image only when user explicitly opts into visualization mode; any shot flagged for heavy iteration is extracted to a standalone text storyboard node) → video-model choice card (H3 default, Seedance 2.0 fallback) + resolution choice card → single-shot clips rendered by the chosen video model → full-film assembly with BGM; final video must remove all storyboard artifacts**.

## Global Visual Style Lock

Unless the user explicitly requests another visual style, all character cards, scene cards, shot tables, text storyboards, optional pencil storyboards, single-shot video clips (regardless of which video model is selected), assembled videos, and final composites must use this visual style:

- Rendering style: Pixar-inspired 3D cartoon rendering, C4D + Octane renderer look, high-end animated feature quality.
- Character design: exaggerated geometric simplification balanced with excellent material detail. Avoid 100% realistic human anatomy; use high-level shape language, large readable silhouettes, and Q-version proportions when appropriate.
- Proportion language: friendly stylized proportions, often 2.5–3 head-tall for cute or childlike characters, with big heads, compact bodies, clear silhouettes, and high recognizability.
- Hair / fur: combine strong sculpted clumps and clean block shapes with fine edge flyaway hairs or fuzzy rim details, so hair/fur feels designed but tactile under light.
- Skin / material: warm subsurface scattering skin quality, soft translucent reddish light through ears, cheeks, nose, and fingertips; avoid hard plastic skin.
- Acting style: exaggerated, lively Disney/Pixar-style character animation performance with squash and stretch, strong brows, eye corners, pupils, lips, and cheek shape changes.
- Motion style: high-energy poses, clear line of action, forward lean, strong anticipation, fast but readable timing, elastic body mechanics, and vivid micro-expressions.
- Emotional range: balance cuteness and explosive expressiveness; intense emotions may use dramatic facial deformation while preserving character appeal.

Negative style constraints: no photorealistic live-action, no flat 2D anime, no plastic toy skin, no stiff mannequin posing, no realistic anatomical stiffness, no lifeless facial expressions.

## STEP 0: Intake and Canvas Plan

Capture:

- One-line idea or rough premise
- Desired output: blueprint only, assets only, standardized shot table with per-second directives, single text storyboards document (default) + extracted single-shot text storyboard nodes for heavy-iteration shots + multi-panel pencil storyboards (opt-in), single-shot video clips (with video model chosen in Step 7), assembled main video, or final BGM-composited film
- Approximate length, if the user already stated it
- Screen size / aspect ratio, if the user already stated it
- Visual tone: default warm stylized 3D animation
- Dialogue requirement: whether the film has dialogue, voiceover, or no speech
- Dialogue language only if the user explicitly states it; do not default to English dialogue

Immediately after capturing the user's input, before Project Brief or any other next step, show choice cards to confirm production format:

Screen size / aspect ratio card:

- 16:9 landscape (recommended for cinematic short)
- 9:16 vertical short
- 1:1 square
- 4:5 social portrait
- Custom size / aspect ratio

Total duration card:

- 30–60 seconds (recommended)
- 15–30 seconds
- 60–90 seconds
- 90–180 seconds
- Custom duration

Only proceed after the user chooses both screen size/aspect ratio and total duration, or explicitly supplies custom values. Store the approved screen size/aspect ratio and duration in the Project Brief and reuse them in shot timing, transition continuity, the standardized shot table with per-second directives, single-shot storyboards (text by default, pencil image when user opts in), single-shot video clips, assembly, BGM matching, and final composite settings.

Create or later update canvas artifacts in this order:

1. Project Brief text node
2. Story Outline text node
3. Labeled Character Card image nodes
4. Environment-only Scene Card image nodes
5. Standardized Shot Information Table node with six columns; each row must include per-second directives inside `Shot Description`
6. **Single text storyboards document** — one canvas text node named `<title> text storyboards` containing one section per shot (mirrors the half-narrated-drama storyboard structure). When the user flags a shot for heavy iteration, that section is extracted to a standalone text node and a `(extracted)` marker is left in the document. Pencil image storyboards, if opted in, are separate image nodes.
7. Single-Shot Video Clip nodes (rendered by the video model selected in Step 7 — H3 default, Seedance 2.0 fallback)
8. Assembled Main Video node
9. Matched BGM audio node and Final BGM-Composited Video node

Do not dump long production content only in chat. Put durable outputs on canvas as text, image, video, or audio nodes.

## STEP 1: Project Brief

Produce a concise project brief and write it to a canvas text node named with the project title or `项目简报`.

Include:

- Working title
- One-line What-if
- Emotional premise
- Target audience feeling
- Main deliverables planned
- Approved screen size / aspect ratio
- Approved total duration
- Dialogue mode and language: only use a specific language when the user explicitly requested it; otherwise write `language not specified` and keep dialogue minimal or ask later when needed
- Initial risks
- Dialogue intent when present

Then show a user choice card:

- Continue with this direction (recommended)
- Regenerate premise options
- Revise emotional premise
- Refine dialogue direction

Only proceed after the user chooses or explicitly says to continue.

## STEP 2: Story Outline and Gates

Create a story outline and write it to a canvas text node named `故事大纲` or `story-outline`.

Include:

- Protagonist Want / Need / flaw
- Core world rule
- 8-beat causal story spine
- Emotional anchor and payoff
- Dialogue beats if the user requested dialogue
- Red-line checks

Gate checks:

- Protagonist is active
- Crisis is intensified by protagonist flaw
- Coincidence never solves the problem
- Ending reuses an earlier emotional anchor
- Antagonistic pressure is not a flat villain
- Dialogue reveals relationship change instead of explaining the theme

Then show a user choice card:

- Approve story and continue (recommended)
- Revise beats
- Revise emotion curve
- Revise dialogue beats
- Return to premise

## STEP 3: Character Cards

Generate character reference cards and place each image on canvas. Recommended order:

1. Protagonist card
2. Contrast / pressure character card
3. Optional supporting character card

Each character card should be a 16:9 production reference sheet when possible. Unlike final rendered video, character cards should include clear readable labels so downstream generation can bind the correct person and props:

- Character name label in English and/or the project language
- Role label, such as protagonist, grandma, thief, sidekick, pressure character
- Main 3/4 view
- Front / side / back views
- Expressions
- Material / costume / prop details
- Important prop labels, such as handbag, wallet, skateboard, apple basket, scarf, shoes, glasses
- Identity lock repeated in the prompt
- A short visual-ID note listing age range, body type, hairstyle, outfit colors, signature props, and do-not-change traits

For stylized 3D animation, keep the character soft, readable, and consistent across later images and videos.

After the main character cards are generated, show a user choice card:

- Lock character designs and continue (recommended)
- Regenerate protagonist card
- Adjust specific visual details
- Add another character card

Warn the user that changing locked character designs later may require regenerating the shot table, single-shot storyboards, single-shot video clips, assembled main video, and final composite.

## STEP 4: Scene Cards

Generate scene reference cards and place them on canvas after character cards. Scene cards must show environments only: do not include characters, people, crowd figures, silhouettes, hands, faces, or character cameos. Character action belongs in the shot table, single-shot multi-panel pencil storyboards, and single-shot video clips, not scene cards.

Include:

- Main environment overview
- Key light states, such as day / night
- Emotional sub-spaces
- Continuity landmarks (fixed objects whose screen position must persist across shots in the same scene, e.g. kitchen island, sofa, door frame, tree, mailbox)
- Important props in the environment

Then show a user choice card:

- Lock scene design and continue (recommended)
- Regenerate scene card
- Add another scene angle
- Adjust lighting or layout

## STEP 5: Standardized Shot Table Video Prompts (Six Columns)

After character cards and scene cards are locked, output standardized video prompts as a shot information table. This step is mandatory and cannot be swapped with storyboard or video generation.

Required reference: read and follow `references/shot-table-spec.md` for the exact six-column schema, per-second directive requirements, table-wide rules, user approval card, and mandatory Step 5.5 self-check gate.

Minimum runtime contract:

- Create a canvas table node named `标准镜头信息表` or `standard-shot-table`.
- Use exactly six columns: `Shot ID & Duration`, `Continuity Handoff`, `Reference Anchors (Spatial + Identity)`, `Hook Type`, `Shot Description (Per-Second Directives)`, `Audio & Dialogue Track`.
- Every row must include complete per-second directives, continuity handoff, reference anchors, hook type, and audio/dialogue timing.
- Run the Step 5.5 self-check from `references/shot-table-spec.md` before storyboarding. Do not enter Step 6 until the self-check passes.

Then show the table approval/self-check choice cards defined in `references/shot-table-spec.md`.

## STEP 6: Text Storyboards Document (Default) + Pencil Image Storyboards (Opt-in)

After the Step 5.5 self-check passes, show a storyboard-mode choice card before producing any storyboard artifact.

Required reference: read and follow `references/storyboard-guidelines.md` for the default single text storyboards document, optional multi-panel pencil storyboards, shot-level extraction/re-integration, storyboard approval cards, and visualization fallback rules.

Minimum runtime contract:

- Default mode is one authoritative text storyboards document containing one section per shot.
- Pencil storyboard images are opt-in visualization artifacts only; they never override the text storyboard.
- Extract a shot into a standalone text node only when the user flags that shot for heavy iteration.
- Step 7 must read the matching text storyboard section or extracted standalone node, not the pencil image.

After all storyboards are approved, proceed to the video-model choice card.

## STEP 7: Video-Model Choice Card + Single-Shot Video Clips

Before any clip is rendered, show the video-model choice card and resolution choice card.

Required references:

- Read `references/model-selection.md` for the H3 default, Seedance 2.0 fallback, per-shot mixed mode, resolution choices, and model-specific prompt shaping.
- Read `references/fallback-policy.md` for per-model retry ladders, drift handling, and escalation choices.

Minimum runtime contract:

- H3 is the recommended default model.
- Seedance 2.0 is the fallback for high-stakes animation performance or repeated H3 failure.
- Per-shot mixed mode is allowed only when the shot table marks the model per row; unmarked rows default to H3.
- Strip all storyboard-only labels before video render.
- Bind each clip to the approved text storyboard section, exact character cards, and exact scene card.
- If a clip drifts from the approved `Reference Anchors`, follow `references/fallback-policy.md`; do not silently assemble incorrect clips.

After all clips render, place them on canvas in shot order, group them as `<title> shot clips`, and show the clip approval card defined in `references/model-selection.md`.

## STEP 8: Full Film Assembly, BGM Match, and Final Output

After all single-shot clips are approved, assemble the complete main video, match or generate one continuous BGM track, and produce the final composited video.

Required reference: read `references/qc-checklist.md` for assembly rules, BGM rules, final review checks, canvas ordering/grouping discipline, user choice-card discipline, and regeneration/latest-asset discipline.

Minimum runtime contract:

- Preserve the exact shot order from the approved table.
- Use only approved latest assets.
- Duck BGM under dialogue, reactions, and important SFX.
- Do not add subtitles or text unless explicitly requested.
- Final video must contain no storyboard traces, labels, arrows, timing marks, panel borders, or double-binding labels.

Then run the final review checks from `references/qc-checklist.md` and deliver the final approved asset.

## Boundaries

Do not use this Skill for a single image, a simple edit, a single clip animation, logo design, or pure prompt consultation. If the user only wants a prompt, use a video prompt workflow instead. If the user only wants a character card, use a character breakdown workflow instead.

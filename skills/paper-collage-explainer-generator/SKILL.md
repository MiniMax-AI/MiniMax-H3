---
name: paper-collage-explainer-generator
description: |
  For creators, educators, and social-video editors who need a tactile paper-collage language for narration, knowledge points, opinions, or abstract topics. Users provide source copy, story beats, or a core concept and may specify aspect ratio, duration, palette, and audio needs. The Skill extracts meaning, proposes visual metaphors, prepares a production plan and storyboard, generates approved halftone collage stills, then creates stop-motion clips with paper movement and tactile sound effects, with optional final assembly. By default it keeps collage SFX and does not add BGM, voiceover, or subtitles unless requested. Best for explainers, viewpoints, story visuals, and social B-roll; not for presenter ads, editable layers, complex typography, or prompt-only tasks.
trigger-words: [paper collage explainer, paper-collage animation, halftone collage, collage explainer, 纸拼贴, 拼贴科普, 定格拼贴, 拼贴动画]
exported-by: MiniMax-hub
---

# Paper Collage Explainer Generator

Turn a short narration line, story topic, viewpoint sentence, or abstract idea into a cohesive editorial paper-collage animation sequence. The visual language is premium halftone paper collage: flat bold color fields, black-and-white photographic cut-outs, selective colored cardstock accents, warm cream keylines, soft paper shadows, tactile stop-motion assembly, and crisp collage sound effects.

This Hub-adapted Skill uses Hub-native image, video, audio, and optional postprocess capabilities. It prioritizes style continuity, color harmony, controlled paper texture, stop-motion collage rhythm, and an audio policy that keeps tactile collage SFX by default while explicitly not adding BGM, voiceover, or subtitles unless the user requests them.

## When to Use

Use this Skill when the user wants:

- A short script line turned into a visual-metaphor collage animation
- A simple story or literary topic explained through collage B-roll
- Editorial paper-collage animation for narration or social video
- Halftone collage animation with objects assembling from an empty color field
- A batch of short abstract sentences or story beats converted into separate visual-metaphor animation clips
- A tactile knowledge explainer that may include paper clicks, pops, slides, presses, and rustles but not automatic music or voiceover

Do not use this Skill when the user needs:

- A realistic product ad or presenter-led video
- Precise editable layers, timeline keyframes, or transparent cut-out assets
- Exact logo placement or readable typography
- Only a written video prompt without generation

## Default Creative Targets

Unless the user specifies otherwise:

- Output ratio: 16:9 landscape
- Clip length: about 4 seconds per segment
- Audio default: keep or generate tactile paper-collage sound effects only, such as paper slide, pop, press, light rustle, and soft tap sounds
- Default: do not add BGM. You may ask whether the user wants BGM when it may help, but add music only after explicit user confirmation
- Default: do not add voiceover, spoken narration, or presenter narration. You may ask whether the user wants narration/口播 when the project may benefit, but write, synthesize, or add spoken audio only after explicit user confirmation
- Default: do not add subtitles. You may ask whether the user wants subtitles, but create or burn in subtitles only after explicit user confirmation
- Visual style: premium editorial halftone paper collage
- Motion style: tactile stop-motion assembly, not slow zoom, generic drifting, or smooth digital layer movement
- Default video generation model: `MiniMax-H3`, unless the user explicitly specifies another model, the model is unavailable, or a hard capability requirement excludes it
- Text in image/video: avoid readable letters, numerals, UI, subtitles, watermark, and logos
- Image quality and depth: prioritize visually attractive, layered 16:9 compositions with clear foreground, midground, and background depth, strong subject hierarchy, rich but readable scene design, and controlled negative space

## Audio Policy

This Skill's default delivery is **with collage SFX, and without BGM, voiceover, or subtitles**.

1. During the first production-plan confirmation, state the default media approach explicitly: tactile paper-collage sound effects are kept or generated; BGM, voiceover/口播, and subtitles are not added by default.
2. It is acceptable to ask the user whether they want **voiceover narration/口播**, **BGM**, or **subtitles**, especially for explainers, but present all three as optional add-ons, not defaults.
3. Do not infer that an explainer automatically needs spoken narration. If the user does not choose narration, create visual story beats rather than a voiceover script.
4. Do not infer that a social video automatically needs BGM or subtitles. If the user does not choose them, keep the clip SFX only.
5. When generating video clips, request synchronized tactile collage SFX if the selected video model supports audio: paper pieces sliding, popping, pressing flat, soft taps, and light paper rustles.
6. During final assembly, preserve the original clip audio tracks when they contain collage SFX. Do not drop audio by default.
7. Remove or replace audio only when the user asked for silence, when the generated audio contains unwanted speech/music, or when the user requests a separate music/narration mix. Create subtitles only when the user explicitly asks for them.

## Global Style Rules

Apply these rules to every still and video in the project:

1. **Unify the overall visual style.** Every segment should feel like part of the same editorial paper-collage series: halftone cut-outs, flat color fields, warm cream keylines, soft physical shadows, clean composition, tactile paper material, and coordinated collage SFX.
2. **Control the paper texture intensity.** Paper must not look perfectly flat, but it also must not become over-aged, dirty, wrinkled, or brown unless the user explicitly asks. Prefer clean, refined hand-made texture: subtle fiber, slightly irregular torn edges, light deckled fibers, layered seams, and soft shadows.
3. **Unify the color tone.** Do not introduce kraft-paper, brown, yellowed, or distressed base papers when they clash with the approved still or the batch palette. Start each clip from a paper field that matches the approved final still's main color direction.
4. **Make motion read as stop-motion collage.** Use clear paper-piece actions: appear piece by piece, slide or pop into position, lightly bounce, press flat, pause, then lock into the final composition. Avoid fast spinning, excessive flipping, chaotic object flight, global fades, smooth digital panning, zooming, or generic drifting.
5. **Keep segments coordinated.** Once one or two clips establish the approved batch style and SFX cadence, later clips and revisions should reference that cadence and tone so the whole sequence feels coherent.
6. **Default to polished 16:9 layered scenes.** Unless the user explicitly requests another platform frame, plan and generate in 16:9 landscape. Use the wider canvas to build attractive foreground / midground / background separation, richer environment props, clear subject hierarchy, and cinematic lateral composition without turning the frame into clutter.
7. **Emphasize paper-collage craft.** Every still and clip should make the paper-collage method visible: separable paper groups, halftone photographic cut-outs, colored cardstock accents, tactile shadows, torn edges, layered seams, and stop-motion actions such as pop-in, slide-in, light bounce, press-flat, pause, and lock.

## STEP 1: Parse the Input

For each line, concept, story, or topic, extract:

- Core meaning: what the viewer should understand
- Emotion: calm, urgent, ironic, surprising, absurd, clarifying, reflective, mysterious, or playful
- Action verb: open, connect, leak, archive, compress, split, illuminate, bind, assemble, reveal, fall, chase, transform, collide
- Visual metaphor: a concrete image that expresses the idea without writing the script on screen
- Key objects: three to six large readable paper groups
- Audio implication: whether the beat benefits from paper slide, pop, press, tap, rustle, or snap SFX

For a story topic, split the story into three to six concise beats unless the user specifies a count. Similar beats may share the same design language, but each should have a distinct metaphor, object set, color field, and SFX rhythm.

## STEP 2: Gate 1 — Production Plan Document Approval

Before generating any media, create a concise production plan document and stop for user approval. Do not generate stills or videos before the user confirms this document.

The production plan must include:

### Brief

- Topic or source line
- Intended audience / use context when known
- Aspect ratio and duration assumptions
- Tone and pacing
- Visual style summary
- Media approach, stated as: default collage SFX are kept or generated; BGM, voiceover/口播, and subtitles are not added unless the user explicitly requests them
- Optional add-on question when useful: ask whether the user wants voiceover narration/口播, BGM, or subtitles, but keep them optional

### Visual Metaphors

For each planned segment:

- Core meaning
- Emotion
- One-sentence visual proposition
- Three to six key objects
- Suggested background color and accent colors
- Expected assembly order
- Expected collage SFX moments, such as slide, pop, press, tap, rustle, or snap

### Script / Visual Beat Track

If the user explicitly requests narration, write a concise voiceover script. If the user has not explicitly requested narration, do **not** write a voiceover script; instead write a silent visual beat track explaining what the audience understands from the sequence.

If the user only needs B-roll for an existing line, preserve the original line as context instead of inventing narration.

### Storyboard

For each segment, include:

- Segment title
- Final-frame description
- Motion idea
- Approximate duration
- Collage SFX idea
- Notes on style continuity and color harmony

After presenting the production plan document, wait for the user to approve, reject, or revise. If the user approves only some numbered items, move only those items forward and revise the rest.

## STEP 3: Build Still-Frame Specifications

After the production plan is approved, write a compact visual specification for each approved segment. The specification should be self-contained and suitable for image generation.

Include:

- Script meaning or story beat
- Visual metaphor
- Aspect ratio
- Background color field
- Accent colors
- Key object groups and their roles
- Composition, foreground / midground / background depth, and negative space
- Final frame relationship
- Style continuity notes
- Avoid list for this specific still

Use this style signature:

```text
flat bold color field, black-and-white halftone photographic cut-outs, selective colored cardstock accents, warm cream keylines, soft paper shadows, fine uncoated-paper grain, premium editorial paper collage, clean refined hand-torn paper edges, subtle fibrous edges, layered paper seams
```

Color guidance:

- Burnt orange or red: labor, time pressure, urgency
- Mustard yellow: tools, warning, accumulated errors
- Ink green: cognition, reset, judgment, surreal calm
- Deep purple: memory, structure, mystery, dream logic
- Teal: collaboration, execution, system flow
- Rose red: absurdity, ceremony, theatrical tension

Do not make cobalt blue the automatic default. Keep the batch visually unified through paper texture, halftone treatment, keylines, shadows, framing, and a controlled palette. Vary the color field only when the story beat benefits from contrast.

## STEP 4: Gate 2 — Generate and Approve Still Frames

Generate one final still frame per approved segment. The still must look like the completed last frame of the future animation.

Still-frame requirements:

- Use the user-specified aspect ratio; otherwise default to 16:9 landscape
- Flat bold paper background with subtle fiber texture
- Three to six large separable object groups, unless the approved storyboard requires slightly more
- Clear central subject, strong visual hierarchy, generous clean negative space, and visible foreground / midground / background layering
- Black-and-white halftone photographic cut-outs as the main structure, combined with rich but readable scene props that support the story beat
- Selective colored cardstock accents only where they clarify hierarchy
- Warm cream keylines, soft physical paper shadows, refined torn edges, and subtle layered paper seams
- No readable text, fake letters, numerals, subtitles, UI, watermark, or logos unless explicitly requested

Show the generated still frames to the user and stop for approval before video generation. If a still looks too busy, has mismatched color, too many paper layers, too-flat digital edges, or too much aged/brown paper texture, revise the still before video generation.

## STEP 5: Plan the Stop-Motion Assembly

For each approved still frame, prepare a motion plan that treats the approved still as the completed final frame.

Default motion order:

1. Start from a clean paper color field matching the approved still's main background, not a mismatched kraft-paper or brown base.
2. Base structure appears as paper pieces that slide, pop, or press into place.
3. Character, card, object, or main metaphor element enters.
4. Secondary objects assemble one by one with small bounce, press-flat, and pause beats.
5. Final relationship locks into the approved composition.
6. Hold the completed still-like frame briefly at the end.

Motion prompt guidance:

```text
Paper-collage stop-motion assembly. Start on a clean paper field matching the approved still's background color and tone. Assemble the scene piece by piece with tactile stop-motion timing: foreground, midground, and background paper groups appear in a readable order; each paper object appears, slides or pops in, lightly bounces, presses flat, pauses, and locks into place. Preserve the halftone dots, cream keylines, subtle fibrous torn edges, layered paper seams, soft shadows, paper grain, color field, and approved aspect ratio. End by holding the completed composition matching the approved still frame. Sound design: tactile collage SFX only, synchronized to paper-piece motion: soft paper slide, pop-in, press-flat tap, light rustle, and tiny paper snap where appropriate. Default: do not add BGM, voiceover, dialogue, or subtitles unless the user explicitly requested them. No mismatched kraft-paper opening, no brown/yellowed base unless approved, no fast spinning, no chaotic object flight, no smooth digital layer movement, no scene cuts, no camera move, no zoom, no morphing, no new objects, no text, no logos, no watermark, no UI.
```

## STEP 6: Generate Collage-SFX B-roll Clips

After Gate 2 approval, generate one video clip per approved still frame.

Generation requirements:

- Use the approved still as the visual reference / final-frame anchor whenever the selected Hub video model supports it
- Use the same ratio as the still frame
- Target about four seconds unless the user requested a different length
- Generate or preserve tactile collage SFX by default when supported: paper slide, pop, press-flat tap, light rustle, tiny snap
- Default: do not generate or add BGM unless the user explicitly requested music
- Default: do not generate or add voiceover, spoken narration, or presenter audio unless the user explicitly requested narration/口播
- If a generated clip contains useful collage SFX, preserve it during post-production and final assembly
- If a generated clip contains unwanted speech, music, fake UI sound, or noisy ambience, remove or regenerate the audio according to the user goal
- Keep the shot visually stable; do not add camera movement unless the user asks
- Preserve the collage material language, final composition, and established batch style
- When revising later clips, use the best approved previous clips as style/cadence references if helpful

Use `MiniMax-H3` as the default video generation model for this Skill. Do not ask the user to choose a model unless `MiniMax-H3` fails, is unavailable, or cannot satisfy a hard capability requirement. If the user explicitly specified another model, keep that model as primary and adapt parameters only when needed.

## STEP 7: Quality Review

Review stills and clips against these standards:

- The metaphor or story beat is understandable without writing it on screen
- The still has a strong clean color field, attractive 16:9 composition by default, clear object hierarchy, and visible foreground / midground / background depth
- The number of object groups stays readable, not a screen full of fragments
- The clip begins from a color field matching the approved still or batch palette
- The assembly is visible piece by piece, not a global fade-in
- Motion reads as stop-motion collage: separate paper groups pop in or slide in, lightly bounce, press flat, pause, and lock
- Defaults: tactile collage SFX are desirable; BGM, voiceover/口播, and subtitles are absent unless explicitly requested
- Paper texture is controlled: not too flat, not over-aged, not brown/kraft unless approved
- Color tone is consistent across the batch
- The final frame remains close to the approved still
- There are no unwanted readable letters, fake UI, subtitles, watermark, or logos

Minor drift in tiny details is acceptable if the metaphor remains clear and the final composition still resembles the approved still. Major drift, added text, fake UI, mismatched opening color, unwanted kraft-paper base, over-wrinkled/dirty paper texture, lost stop-motion assembly, unwanted BGM, or unwanted voiceover should be regenerated or post-processed depending on where the issue appears.

## STEP 8: Optional Assembly, Music, Narration, and Delivery

If the user asks for a multi-clip story video, assemble the approved clips in the confirmed storyboard order and preserve each clip's original collage SFX by default.

Add BGM, narration/口播, or subtitles only when explicitly requested. When the user requests BGM or narration after clips already exist, mix it with the collage SFX instead of replacing SFX, unless the user asks to mute the original SFX.

For each completed item, deliver:

- Final B-roll video path or final assembled video path
- Approved still frame path when relevant
- A one-sentence explanation of how the script or visual beat became a visual metaphor
- Media note: collage SFX by default, and list any explicitly requested BGM, narration/口播, or subtitle additions
- Any QA caveat if a clip passed with minor acceptable drift

For batch work, group the outputs by item number and keep the user's original line or story beat attached to its result.

## Failure Handling

- If the production plan defaults to silent video, correct it to collage SFX only unless the user explicitly asked for silence.
- If the production plan includes BGM, voiceover/口播, or subtitles without explicit user request, remove them and present BGM, voiceover/口播, and subtitles as optional choices.
- If the production plan feels too literal, return to Gate 1 and make the visual proposition more object-driven.
- If the still contains fake text or UI, regenerate the still before video generation; do not try to fix it only in the video prompt.
- If the still or clip is too visually busy, reduce paper layers, simplify the object count, and restore negative space.
- If the video lacks assembly motion, reduce the number of objects and make the step-by-step arrival order more explicit.
- If motion feels like smooth digital layers, rewrite the prompt around stop-motion beats: appear → light bounce → press flat → pause → lock.
- If the paper looks too flat, add subtle torn edges, fiber texture, layered seams, and soft shadows.
- If the paper looks over-aged or dirty, remove kraft-paper, brown/yellowed base, heavy wrinkles, stains, and distressed treatment.
- If the opening color clashes with the approved still, start from the same main background color as the still.
- If the final frame drifts too far from the still, strengthen the instruction that the approved still is the completed final composition.
- If generated audio lacks collage SFX but the visual is otherwise strong, ask the user whether to keep it, regenerate with stronger SFX direction, or add/post-sync SFX.
- If generated audio contains unwanted BGM or voiceover, remove or regenerate it before final delivery.
- If precise layer control is required, recommend a layered animation or editing workflow instead of continuing this Skill.

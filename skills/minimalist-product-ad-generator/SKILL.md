---
name: minimalist-product-ad-generator
description: |
  Turn product images and ad requirements into minimalist product ad shorts for e-commerce promotion and product launches. The Skill confirms format and product variants, extracts selling points, writes concise English ad copy, builds product anchors, plans beat-synced typography/storyboards, and generates a clean product film with premium camera language. Not for KOC talking-head ads, general editing, or complex screen demos.
metadata:
  trigger-words: [minimalist product ad, premium product ad, minimalist product film, 极简产品广告, 高质感产品广告, 产品广告片, 电商产品视频, 新品发布广告]
---

# Minimalist Product Ad Generator

Use this Skill to guide non-professional users through a flow-style workflow that creates a minimalist product advertising video for e-commerce promotion and product launches. The target users are e-commerce sellers, small brand owners, indie creators, and individual sellers; the user should provide at least one product image or related asset.

The core principle is: **confirm assets and brief first, build product facts and a product narrative spine, lock product visuals with independent anchor photos, control video with a precise beat storyboard, then finish the film with native audio or music-based editing.** This Skill no longer defaults to a 4-panel anchor sheet, because video models may reproduce the panel layout. By default, it uses three separate anchor photos plus one precise beat text storyboard as the video control system.

## Start Gate

Every time this Skill is triggered, run a complete but lightweight start gate before any analysis, copywriting, anchor generation, or video generation. Confirm only the information required for production. Do not ask about music here; music is handled later in the music step.

The start gate must confirm all of the following in one pass:

1. **Product images and related materials**
   - Ask the user to upload product images and related materials, such as product photos, packaging images, renders, detail shots, existing video assets, or visual references.
   - If the current message or current canvas / session already contains materials, state that materials have been detected. Do not ask the user to upload again; only confirm whether to use the current materials.
   - If there are multiple candidate materials and their roles are unclear, ask which one should be the main product image.

2. **Product variant status**
   - Confirm whether this is a single variant or multiple variants / colors / models.
   - If there are multiple variants, confirm which one is the main variant.
   - If the user does not specify, the agent may recommend one based on the product image, but the later storyboard must still write the concrete color or variant name.

3. **Target duration**
   - 5s / 10s / 15s, with 10s as the recommended default.

4. **Aspect ratio**
   - 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / match reference image.
   - Once selected, the anchor sheet, video, and final delivery must strictly keep this ratio.

5. **Apple-style template**
   - White-tech / dark rim light / brand color field / light lifestyle scene.

6. **In-frame copy**
   - User-provided copy, or agent-generated Apple-style English copy.

7. **Video model strategy**
   - Default model is MiniMax-H3, because it often produces cleaner Apple-like product close-ups, negative space, and premium product-film camera language.
   - Do not ask the user to choose a video model during the start gate. Show “default model: MiniMax-H3” as part of the adopted start-gate result.
   - If the user explicitly asks for another model, follow the user's model choice after capability checking.
   - Seedance is only a backup when the user explicitly wants stronger multi-shot execution, asks for Seedance, or accepts a fallback after H3 fails or cannot satisfy the requested structure.

Start gate is mandatory. If the user has already uploaded product material, only skip the upload request; do **not** skip the rest of the start gate. The agent must still confirm product variant status, target duration, aspect ratio, Apple-style template, and in-frame copy mode in one pass before analysis, anchor generation, storyboard, or video generation. The video model is not a start-gate question: use MiniMax-H3 by default and display it in the completed start-gate result. If the user says “you decide / just do it,” the agent may choose recommended defaults for the remaining fields, but must still display the adopted choices as the completed start-gate result before continuing.

## Operating Principles

1. **Independent anchor photos are the default visual control system**
   - Generate three separate anchor photos by default, not one 4-panel sheet, because video models may copy panel layouts into the final film.
   - The three anchor photos usually cover: hero / striking main view, material or functional detail, and final copy composition.
   - Do not create a separate structure-action anchor when it overlaps with the hero view. Merge structure / action cues into the material-detail anchor unless the product truly needs a distinct mechanism reference.
   - Each anchor photo must be a complete standalone product photo. Do not create grid layouts, split screens, collage boards, framed panels, product walls, or storyboard sheets.
   - The precise beat storyboard determines the actual shot order, action, and text behavior. Do not use anchor photos as direct shot replacements.

2. **In-frame copy must appear as integrated video motion, not a subtitle fallback**
   - Text in the copy anchor photo is only a reference for font, position, single-line layout, and two-part color treatment. It does not mean the whole video can use only that one copy line.
   - The precise beat storyboard may define multiple short English copy lines across shots. Each line must be 3-5 English words, single-line, readable, and follow the two-part color rule: in white-tech style the first half uses black or dark gray, in dark rim-light style the first half may use white, and the second half always uses the specific product color. Do not use isolated 1-2 word feature labels.
   - Video generation must include the storyboard copy inside the frame.
   - The copy does not need to appear in every shot. If the video model tends to make typography mistakes, reduce copy frequency proactively. For a 10-second film, usually keep only one mid-film copy moment plus one final copy moment; do not put text in every shot. The final shot must include one stable single-line copy.
   - By default, ask the video model to generate in-frame typography and its timing directly. Do not silently degrade text into post-production subtitles or reserve empty space for later overlay. Use post-production fallback only when the user explicitly asks for it or when integrated video typography fails and the user accepts a fallback.

3. **Product body color is a hard fidelity constraint**
   - Apple-style means clean composition, premium light, restrained motion, and negative space; it does **not** mean turning the product itself white, silver, or AirPods-like.
   - The product's original body color, accent color, material tint, and visible finish from the user's image must be preserved in all anchor photos and videos.
   - Only the background, lighting environment, and composition may shift toward the selected Apple-style template. Never recolor the product body unless the user explicitly asks.
   - The final copy emphasis color must come from the real product main color, not from a generic silver / white Apple palette.

4. **Every product needs a product-specific narrative**
   - Do not reuse a fixed earbud template or write vague phrases like “product state changes.”
   - The storyboard must be based on product category, form, material, structure, and visible actions.
   - Product motion must be visual and concrete, such as “the lid opens 30 degrees and inner reflection appears,” or “the crown rotates while a highlight slides along the edge.”

4. **Progress must be visible**
   - After each step, show what was produced and what decision is needed next.
   - Do not auto-run the entire pipeline without user confirmation.

## Typography Rules

- In-frame advertising copy must be English.
- If the user provides Chinese copy, translate it into concise English or ask for approval of the English version.
- Visible in-frame copy must be 3-5 English words, preferably no more than 32 English characters including spaces. Do not write isolated 1-2 word feature labels.
- Copy should feel like an Apple product film: sensory, benefit-led, material-led, or a light value proposition; avoid promotional slogans and ecommerce feature-tag wording.
- Font reference: SF Pro Display / SF Pro Text. Prefer `SF Pro Display Semibold` in prompts.
- Use no more than two text colors in a shot.
- In white-tech style, the first half of the text must use black or dark gray; white text is forbidden on white backgrounds. In dark rim-light style, the first half may use white. The second half always uses the specific product color.
- Text must stay on one line. Do not wrap, stack, or split into multiple lines.
- At any moment, only one single-line English copy line may appear in the frame. Do not show two rows of text, two-line titles, title + subtitle pairs, or multiple text blocks at the same time. Two-part typography must continue or replace within the same line, never split into upper and lower rows.
- Do not place text in a lower subtitle position. Prefer the vertically centered visual zone, left-aligned or right-aligned as part of the composition. Text should be slightly larger and feel like a product-film visual element, not an explanatory subtitle.
- Approved mid-film pattern: clean negative space, product motion or product close-up leads the frame, the frame stays uncrowded, and only one single-line English copy appears. Text may sit near the product edge, product surface, or close-up highlight area so it feels integrated into the product image rather than floating like a subtitle. Do not damage negative space or add a second text block just to add copy.
- Default two-part text motion must remain visible: the first half fades in or slides in subtly first, then the second half fades in or slides in subtly later. When the second half appears, the first half shifts gently within the same line to make room.
- The shift should be subtle, around 10-15px or 8-12% of text width. Motion should be smooth and restrained; avoid bouncy, neon, or flashy UI effects.
- Do not make typography completely static just to avoid two-row text. The correct solution is same-line two-part entrance, subtle shift, fade-in, or gentle slide-in; the wrong solution is upper/lower line splitting, two-line titles, or removing text motion.

## STEP 1: Asset Check and Product Fact Summary

If no product material is available, ask the user to upload it. If material is already available, use it and analyze:

- Product category.
- Subject position and frame occupancy.
- Image quality: sharpness, lighting, composition, and visible details.
- Product main color and possible main variant.
- Showable structure: opening, rotating, lifting, snapping, folding, popping, glowing, screen, texture, etc.
- Real appearance features usable for the storyboard: material, edge, button, port, packaging, transparent part, screen, or distinctive silhouette.

If the image quality is not usable, stop and give concrete reshoot advice.

Output a short product fact summary:

- Material used.
- Product category.
- Main color candidates and main variant suggestion.
- Quality pass / fail.
- Showable structure / feature.

## STEP 2: Confirm the Production Brief

Turn the start gate answers into an executable production brief. Do not ask again for parameters that have already been confirmed. The brief drives the narrative spine, copy, anchor sheet, and storyboard.

The production brief must include:

- Product material used.
- Main variant / product main color.
- Aspect ratio.
- Target duration.
- Selected Apple-style template.
- Single-variant / multi-variant strategy.
- In-frame copy mode: user-provided copy, or agent-generated Apple-style English copy.

Multi-variant product rules:

- The user selects a style; do not automatically remove all other variants.
- Every multi-variant project must decide one main variant first.
- Other variants should not all appear at the very beginning; they appear only in process beats, transitions, or the final full set.
- If the user does not specify a main variant, the agent may choose the most visually stable one and use the others for rhythm and layering.
- Avoid ecommerce matrices, nine-grid displays, stacking, scattered layouts, all colors laid flat, or full-screen product walls.

## STEP 3: Choose a Product Narrative Spine

Before copywriting and anchor generation, choose a lightweight product narrative spine. Do not use a broad corporate brand-film structure; this Skill focuses on short Apple-style films for physical products.

If the user has not selected a direction, offer 2-3 concise directions, recommend one, and continue after confirmation. If the user says “you decide,” use the recommended one.

Recommended spines:

1. **Product Launch** (default recommendation)
   - Negative-space opening → main variant establishes hero view → material / structure detail → natural product action → variants or color relation → full-copy closing.
   - Good for earbuds, watches, digital accessories, small appliances, fragrance devices, etc.

2. **Feature Touch**
   - Product stillness → interaction trigger → feature action → detail magnification → result / feeling → closing frame.
   - Good for products with clear opening, rotation, magnetic snap, light change, screen, folding, mist, or similar actions.

3. **Color Family**
   - Main variant appears alone → supporting variants slide in lightly → color order forms → material and structure stay unified → full set closes → copy lands.
   - Good for multi-color, multi-variant, or multi-model products.

Output one sentence describing the selected narrative spine, for example:

> This film uses the Color Family spine: the purple main variant establishes the hero view, other colors enter as supporting layers, and the full product set closes with the English copy.

## STEP 4: Direct the Motion Language

Before copywriting and anchor generation, define the motion language of the film. This step does not generate assets; it defines motion intensity, transition logic, and rhythm peaks for the later storyboard.

Rules:

1. **Transitions are driven by real product or visual elements**
   - Prefer product edges, material highlights, opening / rotation / snapping / sliding actions, geometric changes in variant arrangement, color entry and exit, matched camera direction, or matched product silhouette.
   - Do not use meaningless white flashes, abstract particles, random light effects, or random cuts to fake premium motion.

2. **One main action per beat**
   - Secondary elements should appear with slight delay and should not compete for attention at the same time.
   - Examples: product enters first, then text appears; main variant stabilizes first, then supporting variants slide in; material highlight sweeps first, then text motion begins.

3. **Set strong and quiet moments**
   - 5s film: 1 small peak + 1 stable closing.
   - 10s film: 1-2 peaks + 1-2 braking moments.
   - 15s film: 2-3 peaks + 2 quiet braking moments.
   - Peaks may be product action completion, variant entrance, second-half copy reveal, or full-product closing.
   - Braking moments may be material-detail pause, readable copy hold, or final frame hold.

4. **Keep safe space clear**
   - Product silhouette remains clear, single-line copy stays readable, and the main variant is not blocked.
   - Add a logo only when the user provides one. Do not generate a fake logo.

5. **Avoid fake tech decoration, mirrored white stages, and empty openings**
   - Avoid fake technology interfaces, meaningless glass cards, decorative text walls, unconfirmed metrics, random particles and stacked light effects, one identical easing style across the whole film, and full-screen product walls.
   - White-tech style is not a mirrored white stage and not dead flat white lighting. Do not place the product on reflective floors, glass tables, or cheap studio mirror surfaces.
   - Apple-style white space should be a simple white background; impact comes from striking product angles, real product actions, camera rhythm, and negative-space composition, not from mirror reflections or waiting on empty frames.
   - The opening must not be empty dead time. It should quickly reveal an attractive product action or angle, such as the product rotating smoothly out of its own structure, emerging from an opening mechanism, sliding along its silhouette, or being revealed by an edge highlight. The exact action must be inferred from the current product form, not hardcoded to earbuds.

Output one short “motion language” statement, for example:

> This film uses product-edge highlights and the purple main variant sliding motion to drive transitions. The main peaks are variant entrance and full-copy landing, with a stable final hold for the product and copy.

## STEP 5: Generate or Confirm Apple-style English Copy

Before generating the text anchor and storyboard, obtain a final English copy line for the frame.

Rules:

1. Analyze product category, usage feeling, and narrative spine.
2. Generate 2-3 Apple-style English options, each 3-5 words.
3. Do not use fixed templates or fixed slogans; infer new copy for every product.
4. Avoid promotional language.
5. If the user already has copy, prioritize it. Chinese copy must be localized into concise English.

After the user confirms the copy, proceed to anchor generation. If the user says “you decide,” choose the recommended copy and continue.

## STEP 6: Generate Three Independent Product Anchor Photos

Generate **three independent anchor photos** in the same aspect ratio and resolution as the user-selected video setting. They must be three separate image outputs, not one combined 3-panel or 4-panel sheet. Background follows the selected style, and all three photos must share unified light, shadow, grade, and background language while remaining standalone product photos.

Anchor photo roles:

- Photo 1: Hero / striking main-view anchor, locking the most attractive product view, silhouette, and opening visual direction. Avoid ugly extreme material close-ups as the opening reference.
- Photo 2: Material / functional detail anchor, locking surface material, grain, reflection, tactile quality, and the most important visible mechanism or structure. If the product needs opening, docking, folding, clasping, button, port, lid, screen, or touch cues, merge those cues into this photo.
- Photo 3: Final copy anchor, locking final product composition, font feeling, text position, single-line layout, and two-part color treatment. The copy is a format reference and does not limit the later video to only this one line.

Multi-variant handling:

- The anchor photos must inherit the selected narrative spine and style strategy.
- Photos 1/2 prioritize building the visual system around the main variant.
- Other variants only support rhythm; do not lay everything flat or fill the frame.
- Photo 3 may show the main variant plus a few supporting variants, or a full-set closing composition, but must preserve negative space and order.

Final copy anchor typography rules:

- Must include the Step 5 confirmed copy; do not use placeholder copy.
- Text must stay on one line.
- Font: SF Pro Display Semibold.
- Text color must be split explicitly into two parts: the first half of the copy uses black or white, and the second half uses the specific product main color. If the main color is purple, write “first half black / white, second half purple.” Do not write only “product color + black or white.”
- Photo 3 may contain only this format-reference copy line and no other text; however, later video copy is controlled by the precise beat storyboard and may include multiple short English copy lines.

Show the three independent anchor photos and ask the user to approve, edit, or regenerate a specific photo before proceeding to the precise beat storyboard.

## STEP 7: Precise Beat and Text Storyboard Table

Before generating any video clip, create and read back a precise beat text storyboard table. It is not artwork; it is the execution table for the video model.

The storyboard table must be organized as “User Choice Statement → Beat Content → Principles.”

### User Choice Statement

Write the confirmed style, aspect ratio, duration, narrative spine, main variant, product main color, variant strategy, and copy. Example:

> White-tech style, 16:9, 10 seconds, Color Family spine, purple as the main variant, other colors as supporting layers, copy: Color Meets Sound.

### Beat Content Table

Recommended columns:

| Time range | Shot | Shot purpose | Visual lead | Visual / camera move | Variant state | Copy | Text color | Text effect | Transition / continuity | Rhythm intent |
|---|---|---|---|---|---|---|---|---|---|---|

Table rules:

- 5s suggests 3-4 beats.
- 10s suggests 5-7 beats.
- 15s suggests 6-9 beats.
- Use seconds as time ranges; do not require frame numbers.
- Keep one main action per beat; secondary layers appear with slight delay.
- Clearly define when the main variant appears alone, when supporting variants enter, and when the full set closes.
- Copy splitting may only be described in plain language as first half / second half. Do not write arrows, slashes, plus signs, or separators, because the model may render them as in-frame copy.
- Text color must use the specific product color name and must be split by style: in white-tech style write “first half black or dark gray, second half the specific product color”; only in dark rim-light style may the first half be white. For example, in white-tech style with purple as the main color, write “first half black or dark gray, second half purple.” Do not write “product color + black or white.”
- If typography is error-prone, reduce text beats first: for a 10-second film, usually use only one mid-film copy moment and one final copy moment, with all other beats marked as no text. Do not put text in every shot.
- The final beat must be a stable single full-frame product closing + full single-line copy. Never end on four panels, split screens, storyboard boards, anchor-sheet layout, product windows, or framed grids. Text still belongs in the vertically centered composition zone, left-aligned or right-aligned as part of the frame, not in a lower subtitle position. Add a logo only when the user provides one.

Rhythm intent vocabulary: setup, establish, prepare, impact, brake, settle.

### Principles

1. Keep core text on a single line.
2. At any moment, only one single-line English copy line may appear; forbid two text rows, upper/lower line splits, title + subtitle pairs, or multiple text blocks at the same time.
3. Use exactly two text colors: black or white + confirmed specific product color.
4. The text must appear inside the video; do not rely only on post-production fallback.
5. Shot continuity = element continuity + motion continuity + form continuity.
6. Smoothness = easing + no awkward timing + seamless entrances and exits.
7. Total pacing follows the user-selected duration.
8. Text motion must be visible: the first half fades in or slides in subtly first, then the second half fades in or slides in subtly later. When the second half appears, the first half shifts subtly within the same line to make room. Do not make the text completely static, and do not split it into upper and lower rows.
9. If the style is confirmed, the table must strictly follow that style.

After the user confirms the storyboard table, proceed to video generation.

## STEP 8: Video Generation

Generate video directly from the confirmed three independent anchor photos + precise beat text storyboard table. Video style, background, lighting, mood, and pacing follow the selected style. Aspect ratio and size must strictly follow the user-selected setting; if the user chose 16:9, generate 16:9 and do not auto-change to another ratio.

Video reference rules:

- During video generation, do not pass the original product image to the video model.
- The original product image is used only for product analysis and anchor generation.
- Use the three independent anchor photos to preserve product consistency, hero view, material / functional detail, final composition, and typography layout.
- Creative control comes from the precise beat storyboard: shot order, product action, copy, text motion, continuity, and transitions.
- Anchor photos are only product anchors used to protect identity, body color, material, and layout. Never treat them as the storyboard itself, and never turn the three photos into three sequential video segments.
- The video output must be one continuous full-frame ad film. Do not show four panels, split screens, collage layouts, frames, storyboard boards, grid layouts, or any shot that reproduces an anchor-sheet layout inside the video. This is especially critical for the ending: never return to a grid, storyboard board, small-window montage, or product wall.
- Default to one H3 full-frame product film, not multiple first-frame clips, unless the user explicitly asks for a different construction.
- Video generation defaults to MiniMax-H3 native audio unless the user explicitly asks for another model or silence. Use the later Apple-style tech BGM direction as the audio prompt so picture rhythm and sound are generated together.

The video prompt must include:

- The three independent anchor photos and the role of each photo.
- The full precise beat storyboard table; read every beat and rewrite every beat into the video prompt. Do not summarize only early beats or omit later typography, variants, or closing requirements.
- The exact copy lines. The video prompt must spell out every in-frame copy line verbatim, its time window, first-half / second-half entry timing, colors, and single-line behavior. Do not rely on the storyboard table alone, and do not summarize this as “text beat sync” or “leave space for copy.”
- Prompt completeness check: if the final video-generation prompt does not literally contain every intended English copy line, the task is not ready to dispatch. Rewrite the prompt before video generation.
- Single-line typography hard constraint: at any moment, only one single-line English copy line may appear in the frame; forbid two rows, two-line titles, title + subtitle pairs, or multiple text blocks at the same time. If typography is error-prone, reduce copy frequency instead of putting text in every shot.
- Two-part text motion rules: the first half and second half must continue or replace within the same line, never split into upper and lower rows; motion must be visible, with the first half fading in or sliding in subtly first, then the second half fading in or sliding in subtly later, while the first half shifts gently within the same line. Do not make the typography completely static.
- Shot continuity hard constraints.
- MiniMax-H3 native audio setting and music prompt; native audio is on by default unless the user explicitly asks for silence or another model.
- Do not pass the original product image.

## STEP 9: music-2.6 Tech BGM

Video generation defaults to MiniMax-H3 native audio first, and the Apple-style tech BGM direction below should be written into the video prompt. If H3's native music is unpleasant, too loud, too weak, out of sync, or the user asks for new music, use `music-2.6` to generate a longer standalone instrumental BGM and replace the video audio track. Do not default to ElevenLabs; use ElevenLabs only when the user explicitly requires a strict target duration, and follow the fee-confirmation rule.

Default music direction:

```text
Around 100 BPM, fast tech feeling, block chords, pluck and airy noise bed, kick + sub-bass + sine sweep, wooden percussion, sudden cut-off, within 0.5s everything stops, leaving only the pluck tail to decay.
```

Extended interpretation:

- Techy, Apple-style, product launch feeling.
- Rhythmic drive is allowed, but it must not become cheap EDM.
- Pluck should be crisp, pleasant, and forward; airy noise should feel premium.
- Kick and sub-bass serve edit points; avoid boomy bass.
- Wooden percussion adds tactile texture, not a complex drum loop.
- No vocals, no synthwave, no retro, no gaming, no cheap corporate stock music.

## STEP 10: Music Analysis, Beat-synced Editing, and Delivery Verification

Before final delivery, edit or assemble based on music analysis. Do not simply lay music under the video.

Music analysis:

- Detect rhythm, strong beats, energy sections, and clear single-note hits.
- Find an 8-12 second segment with continuous sound and suitable energy for product display.
- Do not blindly use the first 10 seconds.
- Prefer a segment with a clear entry point, no silence or collapse in the second half, and enough stable time for the closing frame.

Audio assembly rules:

- Keep the video image, cut a continuous music segment, and replace the video audio track.
- Mute the original video track; keep only the new music.
- Keep music loudness restrained: do not maximize loudness, do not use aggressive limiting, and do not raise music too much; but also do not push it so low that the pluck, rhythm points, and product display lose clarity.
- Do not use long fade-outs. Only apply a 0.1-0.3s tail fade when needed.
- After assembly, check for long silence or audible dropouts.

Pre-delivery verification:

- Video aspect ratio, duration, and language match the brief.
- Product appearance, main variant, and variant timing match the anchor sheet and storyboard.
- Copy is readable, single-line, not crowded, and contains no extra arrows or symbols; no two text rows, upper/lower line splits, title + subtitle pairs, or multiple text blocks appear at the same time.
- The final frame has a stable product + copy closing.
- Music is not too loud and not too low.
- Output is on canvas, and multi-asset outputs are grouped.

Delivery includes:

- Final video path or canvas output.
- Duration, aspect ratio, and language.
- Short creative summary.
- Copy used and music segment used.
- Whether beat-synced editing was applied.
- Specific next improvements, such as pacing, copy, main variant, audio, or platform crop.

## Failure Handling

- Product image too weak: stop, give reshoot guidance, and wait for a better image.
- Copy too long: compress to 3-5 English words and ask for approval.
- Text anchor is wrong: regenerate the text anchor before video generation.
- Anchor sheet fails product fidelity: regenerate with stronger product-preservation instructions.
- Storyboard is templated: return to the precise beat table and rewrite based on product form and narrative spine.
- Copy does not appear in video: rerun the video prompt, explicitly requiring copy in at least the middle and final beats.
- Video typography deforms: regenerate the text anchor or degrade the text shot to subtle anchor-based motion.
- Aspect ratio is wrong: immediately rerun with the user-selected ratio.
- Music quality is poor: retry with `music-2.6`, do not force duration, and cut the best segment from the longer track.
- Final assembly has no sound in the second half: re-analyze music, select a continuous audible segment, and use direct audio replacement.

## Trigger Examples

Use this Skill for requests like:

- “帮我的产品做一个苹果味儿广告片”
- “用这张产品图做 Apple 风电商广告”
- “做一个极简高级的产品发布视频”
- “Make an Apple-style product ad from this product photo”
- “Create a premium minimalist ecommerce product film”

Do not use this Skill for general video editing, documentary explainers, KOC talking-head ads, or complex UI/screen-text demos unless the user explicitly wants the Apple-style product ad workflow.

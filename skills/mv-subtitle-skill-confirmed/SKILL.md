---
name: music-video-subtitle-generator
description: |
  For musicians, video creators, and social-media editors producing AI music videos or emotional short films with lyric typography. Users provide music, lyrics, references, characters, typography direction, mood, or target platform. The Skill analyzes beat and vocal timing, separates character, scene, and text references, designs beat-reactive spatial typography, decomposes long works into connected shots, audits prompts, and routes generation for H3 or other video tools. It outputs MV concepts, shot prompts, lyric text plans, and stitching guidance. Best for stylized MVs and subtitle-driven music visuals, not ordinary caption cleanup, licensed IP copying, or fully manual post-production editing.
trigger-words: [MV, music video, lyric typography, on-screen text, prompt audit, Trap MV, Gospel hip-hop, Dark-pop, Cyber-grunge, MV提示词, 歌词文字, 字幕MV, 贴字MV, 卡点MV, 多镜头拼接]
---

# Music Aesthetics MV

## Purpose

Use this Skill when the user wants to create, revise, audit, or generate music-video prompts or emotional short-film prompts where music, lyrics, typography, references, rhythm, performance, and camera language must be designed together. The workflow adapts MV prompt rules into Hub execution: key creative decisions are confirmed, locked prompts are written to canvas text nodes, and media creation is delegated to Hub agents.

Do not use it for ordinary subtitle burn-in, generic video editing, non-music product ads, or simple single-image / single-clip requests without MV structure.

## Hub compatibility rules

- Do not assume third-party tools, shell scripts, browser plugins, or external generation APIs.
- Write locked prompts and revisions to Hub canvas text nodes.
- Delegate character cards, scene cards, typography cards, video clips, BGM, and final assembly to Hub image, video, music, and editing agents.
- Do not hardcode output paths; always use Hub-returned paths.
- If the user says “skip confirmation / just do it”, apply it only to the current run unless the outer orchestrator has active session authorization.
- If the user only wants a prompt, stop after delivery. If the user wants a finished MV, continue to generation and assembly after confirmation.

## Core principles

1. Understand the work before choosing a structure. Templates organize, but are not mandatory forms.
2. Use the latest confirmed creative intent.
3. Omit irrelevant fields instead of filling placeholders.
4. Do not mechanically add performance, typography, transitions, camera moves, or character actions just because a preset includes them.
5. On-screen text is a designed visual layer, not ordinary subtitles, unless the user explicitly requests subtitles.
6. If the user uploads a real song or beat, it is the master music bed unless they request replacement.
7. **Natural multi-shot stitching for >15s videos**: when duration exceeds the model single-generation limit, use “multi-shot storyboard breakdown + tail/head frame continuation + beat hard cuts + global master-audio alignment”. Every segment must preserve the same groove, tempo feel, aspect ratio, character logic, visual preset, lighting language, and typography motion rules.
8. If the user wants a finished music-aesthetic MV without lyrics, generate and lock original lyrics first. The final MV must reference character, typography packaging, and scene cards together. The typography packaging card controls only text packaging style, font texture, graphic design, layout ratio, and motion language.

## STEP 1: Pre-flight lock

Before writing the final production prompt, confirm the minimum foundation with concise recommended choices.

### 1.1 Video format

Offer fixed options:

| Use case | Aspect ratio | Resolution |
| :--- | ---: | ---: |
| TikTok / Reels / Shorts vertical | 9:16 | 1080×1920 |
| YouTube / Bilibili horizontal | 16:9 | 1920×1080 |
| Feed square | 1:1 | 1080×1080 |
| Dense vertical ad test | 9:16 | 720×1280 |
| Cinematic widescreen MV | 21:9 | 2560×1080 |

The selected ratio must be reused across character cards, scene cards, typography references, shot prompts, video clips, and final assembly.

### 1.2 Target duration and multi-shot structure

Before prompt writing or video generation, confirm target MV duration. Never silently use the model default as user intent.

Offer a compact duration card:

1. **10-second test**: one or two shots to verify character, scene, typography, and audio style.
2. **15-second hook**: 2–4 short shots for a more complete chorus / hook / performance phrase.
3. **30-second or longer complete MV (multi-shot stitching)**: because many models have a 15-second single-generation ceiling, automatically use a Multi-Shot stitching workflow. First generate or lock one continuous song / BGM, split 30 seconds into 4–8 dynamic 2–5 second shots, then stitch via beat sync and head/tail frame continuity.
4. **Custom duration**: user provides duration; plan shot count and stitching based on the selected model’s single-clip limit.

If uploaded music exists, the confirmed duration is the music window length.

### 1.3 Music window

If uploaded or specified music is longer than the confirmed duration, lock a segment before writing prompts:

1. Recommended window: assistant identifies the strongest chorus / hook / emotional turn and requests confirmation.
2. User timestamps: user provides start and end seconds; verify both are inside the audio.
3. Multi-variant: only for ad-hook testing or multiple creative directions.

If music is shorter than the target duration, use the full audio unless the user explicitly asks to stretch or extend it.

### 1.3.0 Lyrics lock and lyrics-first fallback

Before final video prompts or generation, determine lyric ownership:

1. If the user provides lyrics, those are locked lyrics. Do not generate, add, rewrite, expand, translate, paraphrase, or replace them unless explicitly requested.
2. If the user wants a finished music-aesthetic MV without lyrics, generate short original lyrics matching the music style, vocal mode, duration, emotion, and visual preset, then lock them.
3. Locked lyrics are the only source text for rap/singing performance and visible typography.
4. Split locked lyrics into each shot’s `Rap line:`, `Vocal line:`, `Soft vocal line:`, or `Spoken line:`.
5. Every `Typography:` field must come from the same locked lyrics. During vocal performance, visible text must word-for-word match the performed phrase.
6. In the final typography MV, the performer must rap or sing the locked lyrics with visible lip shapes, jaw motion, breath, facial accents, nods, and hand accents following phrasing and rhythm.

### 1.3.1 Default multi-shot stitching for >15s videos

For >15s MVs such as 30 seconds, the global multi-shot prompt script is the authoritative source:

1. **Lock complete Master Audio**: lock one continuous song / BGM track with the full vocal and groove as the only audio baseline.
2. **Build a Shotlist Timeline**: split 30 seconds into 4–8 short shots of 2–5 seconds, mapped precisely to lyric timestamps and beats such as snare, 808, and drop.
3. **Head/tail frame and scene continuity**: if Shot B continues the same scene, use Shot A’s tail frame as Shot B’s head frame. If it is a hard cut, preserve the same character card, wardrobe, lighting prompt, and use same-direction camera motion or match-cut visual elements.
4. **Generation and edit assembly**: the video agent generates each shot from the Shotlist. The editing agent aligns all clips to the global Master Audio beat grid, trimming, speed-ramping, and stitching for natural lip-sync, beat sync, and transitions.

## STEP 2: Creative Contract

Before final prompts, create a concise creative contract:

- Music genre, instrumentation, tempo/BPM feel, vocal mode, and emotional temperature.
- Lyric source and locked lyrics.
- Target duration and multi-shot breakdown, e.g. 30s split into 6 short shots.
- Reference image roles: character card, scene card, typography packaging card.
- Camera language, shot sizes, focus, beat-cut density, and transition logic.
- Exclusions such as no fades, no glossy AI beauty face, no single-shot overstretching that causes deformation.

## STEP 3: Reference roles and scene sampling

Assign each reference image one narrow job:

- **Typography reference card**: controls only text packaging style, font texture, graphic design, layout ratio, and motion language. It must not contribute people or scenes.
- **Character reference card**: controls character identity, facial aura, hairstyle, clothing silhouette, proportions, posture, and presence.
- **Scene reference card**: controls scene style, spatial atmosphere, image texture, background depth, and lighting mood.

### 3.1 Close-up scene switching and multi-shot sampling

For Trap, Dark-pop, Cyber-grunge, and other fast-paced MVs, switch quickly among multiple close-up spaces:

- Keep scene mood unified, but each shot’s space must differ clearly, such as local background, tunnel, wall, light strip, or reflective floor.
- Scene switches must be triggered by musical impacts: bass hit, 808 drop, snare, vocal accent, or typography smash.
- Shot switches are “visual samples” on the trap beat: hard cuts, jump cuts, scan-glitch cuts, flash cuts, or action match cuts.

## STEP 4: Preset grammar, Dark-pop / Cyber-grunge and Trap example

### Visual and editing language

- Realistic high-fashion texture and film-magazine texture, referencing late-90s / early-00s indie magazines, photocopy paper, film scans, and zine collage.
- Heavy grain, slight film jitter, halftone dots, rough print edges, scan offsets.
- **Editing must use hard cuts only**. No fades, dissolves, or soft transitions.
- Image and typography strongly respond to beat cues: hi-hat roll creates micro-shake / frame skips, snare triggers scale-up / hard cut / shoulder drop, 808 bass hit creates low-frequency compression / stretch / offset.

### Typography packaging rules

- Typography is a dynamic graphic subject in space, not a subtitle bar. It may sit in foreground, midground, background, or be occluded by shoulders / hands.
- Text must never cover eyes or main facial expression; avoid the mouth during critical lip-sync.
- With vocals, visible words must exactly match the performed lyrics. Each shot has only one main typography event.

## STEP 5: Prompt structure template

A multi-shot script must be modular by shot, with accurate duration and audio mapping:

```text
[Global Aesthetic & Character Lock]: (global character and aesthetic anchor prompt)

Shot 1 (0.0s - 3.5s)
Vocal Line: "..."
Typography: "..."
Visual & Action: ...
Camera & Motion: ...
Transition Out: hard cut to Shot 2 on bass hit / same-direction whip pan

Shot 2 (3.5s - 7.0s)
Vocal Line: "..."
Typography: "..."
Visual & Action: ...
Camera & Motion: ...
Transition Out: ...
```

## STEP 6: BGM continuity and natural multi-shot stitching system

### 6.1 Global audio continuity

The entire MV must bind to one Master Audio track. During segmented video generation, do not use independent disconnected clip audio. All video clips are aligned to their corresponding timestamps on the Master Audio timeline during edit assembly.

### 6.2 Natural Stitching Protocol

For >15s stitching, enforce five continuity locks:

**Vocal & Lip Continuity**: Cut points must land on lyric pauses, breaths, snare, or drop. Do not hard cut inside an active vowel or lyric unless the next shot is an extreme close-up with continuous mouth shape.

**Rhythm & Beat Matching**: Cut points must hit the 1/4 or 1/8 beat grid. Use speed ramping to align head nods, hand gestures, and blinks to beats.

**Aesthetic & Color Grading**: Every cross-shot generation carries the same aesthetic header prompt: grain level, LUT direction, and light direction. Final assembly uses unified 35mm film grain and LUT to hide batch color differences.

**Transition & Motion Continuity**: For long-shot continuation, use the previous tail frame as the next head frame and prompt continuation of action. For hard scene switches, use kinetic continuity such as same-direction pan or hand-occlusion match cut.

**Typography Motion Stitching**: Text in the previous shot should shatter, sweep out, or smash-screen on a bass hit; next-shot text smashes in or unfolds on the accent, transferring visual momentum.

## STEP 7: Checklist

Silently verify before delivery:

- [ ] If target duration exceeds 15s, has Multi-Shot breakdown been used automatically?
- [ ] Is one global Master Audio locked, and are cut points beat-aligned?
- [ ] Do cuts avoid mid-lyric vocal breaks, with natural lip and breath continuity?
- [ ] Do adjacent shots have clear stitching logic such as tail/head continuation, same-direction motion, match cut, or beat hard cut?
- [ ] Is the edit style strictly hard-cut, with no fades or soft transitions?
- [ ] Are character / scene / typography cards isolated without cross-contamination?
- [ ] Is typography a spatial layer that never blocks eyes or main facial expression?
- [ ] Are image quality, grain, color, and lighting consistent globally?

## STEP 8: Canvas delivery

After the complete MV prompt script is locked, write it to a dedicated canvas text node named `Complete MV Prompt` or `完整MV Prompt`. The node must contain the full multi-shot prompt script and stitching notes. Later revisions update the same node.

## STEP 9: Final MV generation and assembly workflow

1. **Prompt lock and canvas write**: confirm the full multi-shot prompt script with timestamps, lyric mapping, and transition logic is written to canvas.
2. **Parallel shot generation**: the video agent generates each 2–5s shot from the Shotlist. For continuation shots, extract the prior shot’s tail frame as the next I2V head frame.
3. **Multi-track editing and stitching**: the editing agent imports the global Master Audio, aligns all shot videos by timestamp, trims and speed-ramps on the beat grid, and checks seams with same-direction motion or flash-cut masking.
4. **Finishing**: apply unified Movie LUT and 35mm film grain overlay to reduce AI plasticity and lock visual consistency.
5. **Delivery**: output the seamless complete MV video file.

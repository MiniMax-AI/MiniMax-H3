# MiniMax H3 Skills

This directory contains the skills bundled with [MiniMax H3](../README.md): **1 prompt writing skill** and **8 style-specific video generation skills**. Each skill lives in its own folder with an installable `SKILL.md` (plus a `SKILL.cn.md` Chinese version for the style skills) and any reference materials it needs.

## Installation

Install skills with the [skills CLI](https://github.com/vercel-labs/skills):

```bash
# List all skills available in this repository
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --list

# Install all skills
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill '*'

# Install a single skill
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

## Skills

### h3-prompt-writing

[SKILL.md](h3-prompt-writing/SKILL.md)

Write structured MiniMax H3 video generation prompts for all five generation modes: T2VA, I2VA, FL2VA, L2VA, and Ref2VA. The skill rewrites multimodal requests into H3's prompt structure — `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` — aligns keyframes, and defines reference labels for images, videos, and audio. It ships with two prompt guides under [`references/`](h3-prompt-writing/references/):

- [`base-en.txt`](h3-prompt-writing/references/base-en.txt) — base text/keyframe modes
- [`ref-en.txt`](h3-prompt-writing/references/ref-en.txt) — full-reference (Ref2VA) mode

### minimalist-product-ad-generator

<p align="center">
  <img src="../assets/minimalist-product-ad-generator.gif" alt="minimalist-product-ad-generator" width="240">
</p>

Turn product images and ad requirements into clean, minimalist product ad shorts for e-commerce promotion and product launches. The skill confirms format and product variants, extracts selling points, writes concise English ad copy, plans beat-synced typography and storyboards, and generates a premium product film with polished camera language. Not for KOC talking-head ads, general editing, or complex screen demos.

[SKILL.md](minimalist-product-ad-generator/SKILL.md) · [SKILL.cn.md](minimalist-product-ad-generator/SKILL.cn.md)

### 3d-animation-short-generator

<p align="center">
  <img src="../assets/3d-animation-short-generator.gif" alt="3d-animation-short-generator" width="240">
</p>

Create complete stylized 3D animated shorts from a story idea through an ordered production workflow: project brief, story outline, character and environment cards, standardized shot planning, text or optional pencil storyboards, video-model selection, single-shot generation, assembly, BGM matching, and final review. Built for end-to-end narrative animation with strong character consistency, scene continuity, timing, camera, performance, and audio control. Not for single images, simple edits, photorealistic live action, or one standalone clip.

[SKILL.md](3d-animation-short-generator/SKILL.md) · [SKILL.cn.md](3d-animation-short-generator/SKILL.cn.md)

### papercraft-stop-motion-explainer

<p align="center">
  <img src="../assets/papercraft-stop-motion-explainer.gif" alt="papercraft-stop-motion-explainer" width="240">
</p>

Explain science, education, or general knowledge through tactile handmade papercraft visuals. The skill extracts the learning goal and visual metaphor, proposes creative directions, designs paper characters, layered diorama sets, and props, creates preview concepts plus image and video prompts, and plans storyboards, camera movement, transitions, and sound with staged approvals and review checklists. It outputs a production-ready papercraft stop-motion explainer package, or selected assets such as still prompts, image-series prompts, short-video prompts, or storyboards. Best for cut-paper, pop-up-book, layered diorama, and miniature stop-motion explainers.

[SKILL.md](papercraft-stop-motion-explainer/SKILL.md) · [SKILL.cn.md](papercraft-stop-motion-explainer/SKILL.cn.md)

### brand-promo-video-generator

<p align="center">
  <img src="../assets/brand-promo-video-generator.gif" alt="brand-promo-video-generator" width="240">
</p>

For marketers and creators producing promotional content for brands, products, websites, apps, shops, or personal projects. The skill organizes brand facts and asset provenance, selects a narrative direction, plans precise beats and shots, generates needed imagery, video, voiceover, or music, and completes assembly and pre-delivery review. It outputs a promotional short that highlights product capabilities, use cases, and a call to action. Best for launches, website showcases, and social promotion; not for imitating real brand marks without authorized assets or inventing product claims.

[SKILL.md](brand-promo-video-generator/SKILL.md) · [SKILL.cn.md](brand-promo-video-generator/SKILL.cn.md)

### music-video-subtitle-generator

<p align="center">
  <img src="../assets/music-video-subtitle-generator.gif" alt="music-video-subtitle-generator" width="240">
</p>

For musicians, video creators, and social-media editors producing AI music videos or emotional short films with lyric typography. The skill analyzes beat and vocal timing, separates character, scene, and text references, designs beat-reactive spatial typography, decomposes long works into connected shots, audits prompts, and routes generation for H3 or other video tools. It outputs MV concepts, shot prompts, lyric text plans, and stitching guidance. Best for stylized MVs and subtitle-driven music visuals.

[SKILL.md](mv-subtitle-skill-confirmed/SKILL.md) · [SKILL.cn.md](mv-subtitle-skill-confirmed/SKILL.cn.md)

### co-op-game-intro-generator

<p align="center">
  <img src="../assets/co-op-game-intro-generator.gif" alt="co-op-game-intro-generator" width="240">
</p>

Create a two-player co-op game menu or opening animation. The skill locks identity cues, generates an approval image from a fixed menu framework with coordinated color, buttons, icons, and typography, then uses the approved result to rebuild character, UI-copy, and event timing instructions for the final video. It outputs a co-op game intro featuring two characters, player cards, and menu interaction motion. Best for game concepts, character-led menus, and social content.

[SKILL.md](co-op-game-intro-generator/SKILL.md) · [SKILL.cn.md](co-op-game-intro-generator/SKILL.cn.md)

### paper-collage-explainer-generator

<p align="center">
  <img src="../assets/paper-collage-explainer-generator.gif" alt="paper-collage-explainer-generator" width="240">
</p>

Give narration, knowledge points, opinions, or abstract topics a tactile paper-collage language. The skill extracts meaning, proposes visual metaphors, prepares a production plan and storyboard, generates approved halftone collage stills, then creates stop-motion clips with paper movement and tactile sound effects, with optional final assembly. By default it keeps collage SFX and does not add BGM, voiceover, or subtitles unless requested. Best for explainers, viewpoints, story visuals, and social B-roll.

[SKILL.md](paper-collage-explainer-generator/SKILL.md) · [SKILL.cn.md](paper-collage-explainer-generator/SKILL.cn.md)

### handdrawn-live-video-generator

<p align="center">
  <img src="../assets/handdrawn-live-video-generator.gif" alt="handdrawn-live-video-generator" width="240">
</p>

Create surreal short videos that blend rough glowing hand-drawn animation with live-action spaces. The skill clarifies the physical contact, designs continuous morphing, an escape route, and a delayed handheld chase movement, then writes a reusable 15-second 16:9 video prompt in the user's language. After confirmation it recommends MiniMax H3 generation and checks contact realism, camera delay, rough glowing stroke texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene cuts.

[SKILL.md](handdrawn-live-video-generator/SKILL.md) · [SKILL.cn.md](handdrawn-live-video-generator/SKILL.cn.md)

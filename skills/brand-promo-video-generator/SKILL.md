---
name: brand-promo-video-generator
description: For marketers and creators producing promotional content for brands, products, websites, apps, shops, or personal projects. Users provide logos, product images, interface screenshots, official links, or other verifiable assets and confirm duration, aspect ratio, audience, and campaign focus. The Skill organizes brand facts and asset provenance, selects a narrative direction, plans precise beats and shots, generates needed imagery, video, voiceover, or music, and completes assembly and pre-delivery review. It outputs a promotional short that highlights product capabilities, use cases, and a call to action. Best for launches, website showcases, and social promotion; not for imitating real brand marks without authorized assets, inventing product claims, or producing long-form narrative films.
allowed-tools:
- webfetch
- hub_image_search
- hub_analyse_media
- hub_canvas_get_node
- hub_canvas_group_recent_outputs
- hub_generate_image
- hub_generate_video
- hub_generate_audio
- hub_generate_music
- hub_synthesize_speech
- hub_video_edit
- hub_audio_meta
- task
---

# Brand Promo Video Generator

Create a polished short promo video for a brand, product, website, app, shop, or personal project. Use this Skill when the user has a logo, product images, screenshots, a website link, or just a clear idea and wants the agent to turn those materials into a clean brand reel.

This Hub adaptation replaces third-party Vibe Motion / Remotion implementation details with Hub-native orchestration: source research, asset verification, story planning, image/video generation, optional speech or music, and editing assembly. Do not initialize external projects or depend on npm validators during normal execution.

## Tool Coverage Rule

The `allowed-tools` list must cover the full production promise in this Skill: source lookup, image preparation/generation, video clip generation, optional speech/music/audio generation, final editing/assembly, media inspection, and canvas grouping. If a runtime does not provide one of the listed generation or editing tools, downgrade the deliverable explicitly to a pre-production package instead of claiming that a final promo video can be generated.

## STEP 1: Intake assets and resolve the brief

Before any story planning or generation, run a required user intake. Ask the user to upload or provide links to the elements that must be verified:

- Logo files or official logo source pages
- Font files, font names, typography guidance, or official website pages that show brand typography
- Brand colors, color system, style guide, or pages that clearly show official colors
- Product images, UI screenshots, packaging, renders, footage, or other brand imagery
- Product information: official product name, feature list, launch focus, claims, CTA, disclaimers, and target audience
- Company/product official URL or official source package

Treat user-uploaded assets as usable by default for pre-production and concept planning. Do not repeatedly ask a standalone rights/permission question during the first intake unless there is a concrete risk signal, such as a visible third-party watermark, obviously scraped marketplace imagery, contradictory user wording, legal/medical/financial compliance claims, or the user asks for commercial publication. Record the source as "user-provided" and surface rights caveats in the source summary instead of blocking the flow.

In the same opening intake, ask the user to choose:

- Target duration, normally 15-30 seconds; recommend 15 seconds when the user wants a fast launch film
- Aspect ratio; offer common choices such as 16:9, 9:16, 1:1, 4:3, 3:4, or match a supplied reference

Also identify campaign focus, distribution channel, narration language, on-screen copy language, and visible copy needs when they are not already clear. Do not proceed to creative direction until the user has supplied the usable materials or explicitly confirms which elements are unavailable.

Language rule for promo content: choose narration and on-screen copy language from the brand materials, target audience, and platform context, not mechanically from the chat language. If the brand assets and visible source copy are primarily English or global corporate English, default narration and on-screen copy to English unless the user explicitly asks for Chinese localization. If the user is Chinese but says they are testing as a new user, keep chat replies in Chinese, but plan the actual video copy in the language that best fits the brand campaign.

If a logo, product UI, person, mascot, packaging, font, color system, or other identity-bearing asset cannot be authenticated, stop and ask for an authorized original instead of generating a plausible substitute.

## STEP 2: Build the brand truth sheet

Research or inspect the strongest available sources and summarize the brand truth sheet before creative production:

1. User-provided original exports
2. Official company website, static bundle, newsroom, brand portal, media kit, press kit, or official repository
3. Company-controlled media library
4. Licensed stock or authorized partner kit

Extract:

- Exact logo variants, clear space, and usage constraints
- Official fonts or visible typographic behavior
- Primary, secondary, and dynamic brand colors
- Brand tone, principles, visual motifs, and interaction language
- Current product names, features, scenarios, metrics, slogans, CTA, and disclaimers
- Official photography, renders, UI screenshots, footage, press assets, and media kit material

Do not use logo aggregation sites, search thumbnails, fan recreations, Pinterest reposts, or AI-generated substitutes as identity-bearing sources. Procedural graphics are allowed only as non-representational motion layers: masks, gradients, color fields, grids, glows, particles, trails, typography, verified-data charts, and transition geometry.

## STEP 3: Create a provenance manifest

Record every identity-bearing asset in a compact manifest that can be delivered to the user. The manifest may be a text node or table and should include:

- Stable asset ID and role
- Local path or canvas node reference when available
- Exact source URL or user-provided source note
- Source type, such as official website, media kit, user-provided original, or licensed stock
- Verification target used for comparison
- Rights or publication note
- Authenticity status: verified, user-supplied, licensed, or blocked

The manifest does not grant publication rights by itself. If authorization is unclear, label the video as an unofficial concept and tell the user commercial publication requires permission.

## STEP 4: Choose the story spine

Present 2-3 concise creative directions when the user has not already chosen one, recommend one, and continue after confirmation. Use the product category to pick a spine:

- AI / SaaS: user intent -> thinking or planning -> capabilities -> execution -> useful output -> proof -> logo
- Physical product: hero reveal -> interaction -> feature macro -> usage context -> result -> logo
- Service / company: context -> process -> evidence -> outcome -> promise -> logo
- Image-led brand: authentic imagery -> visual motif -> benefit -> emotional payoff -> logo

Keep the story product-specific. Show actual features, interactions, scenarios, outputs, and proof instead of hiding the story behind abstract effects.

## STEP 5: Plan exact beats

Plan a frame-aware timeline before generation. Use 30fps as the planning convention unless the output pipeline requires otherwise.

For a 15-second film, target 5-8 major beats. For a 30-second film, target 8-12 major beats. Each beat should define:

- Start and end time or frame range
- Visual owner and authentic asset IDs
- Primary action
- Product or brand proof shown in the shot
- Copy and readable hold
- Color state
- Incoming and outgoing transition
- Motion intent: setup, anticipation, commitment, impact, brake, settle

A useful 15-second pattern is: brand hook, user intent or setup, product mechanism, capabilities or scenarios, output or proof, product payoff, final logo and CTA. Use 6-12 frame overlaps when outgoing motion naturally supplies the next shot.

## STEP 6: Direct the motion language

Build intensity with control:

- Let product motion, cursor paths, UI flow, light, scrolling content, object edges, or matched geometry drive transitions
- Use 2-5 deliberate color states tied to meaning
- Keep one primary action per beat; delay secondary layers slightly
- Establish 2-3 high-energy peaks and quieter braking moments
- Preserve readable silhouettes, copy, and logo clear space
- Avoid fake HUDs, arbitrary glass cards, decorative text walls, unverified metrics, and identical easing everywhere

For AI products, include at least one readable chain such as: prompt -> planning -> parallel capabilities -> generated result -> proof. For physical or service products, show cause and effect from user action to concrete outcome.

## STEP 7: Hard confirmation before generation

Before generating any video, image sequence, speech, music, or final edit, stop and show the user the completed pre-production package:

- Provenance manifest or source summary
- Brand truth sheet
- Chosen creative direction
- Exact beat / shot plan
- Visible copy, CTA, narration, and audio plan
- Known authenticity, rights, or placeholder caveats

Use a concise confirmation step before generation. If the user clearly expresses approval or intent to proceed after seeing the pre-production package — for example "confirm", "generate", "go ahead", "continue", "next", "可以", "继续", "下一步", or similar — treat it as permission to generate, unless the message also asks for changes. If the user asks to skip the process, still provide a compact source summary, brand truth sheet, and beat plan first, then proceed when they indicate approval. Offer revision choices only when the user's reply is ambiguous or requests changes.

## STEP 8: Produce Hub assets

Use Hub-native generation and editing only after the hard confirmation gate has passed. Keep each dispatch self-contained with the chosen model, aspect ratio, authentic reference paths, and original request.

Typical production flow:

1. Generate or prepare verified still frames, UI plates, product hero frames, or motion-ready story images.
2. Generate video clips from those frames or from precise text prompts, preserving the same ratio and brand assets.
3. Default audio policy for native brand reels: when the user asks for BGM, music, soundtrack, ambient sound, or says nothing beyond needing a finished promo video, prefer video-native audio from the selected video model instead of generating a separate music track. For the default MiniMax H3 route, set native audio on (`generate_audio=true`) and prompt for brand-safe instrumental music / UI sound design inside the video prompt.
4. Generate separate speech or music only when the user explicitly needs controllable narration, voiceover, dialogue, replaceable standalone BGM, exact music duration independent of the video, post-production remixing, or when the selected video model cannot generate suitable audio. Do not duplicate a soundtrack between video-native audio and separate audio generation.
5. Assemble clips, any explicitly separate audio, and final brand lockup in editing. Add subtitles only when the user explicitly asks for subtitles.

Do not redraw or approximate logos, wordmarks, product UI, packaging, mascot, person, or brand scene. Use generated material only for abstract motion, atmosphere, transition geometry, or clearly conceptual scenes that do not impersonate official product evidence.

## STEP 9: Verify before delivery

Before final response, check:

- The logo and identity-bearing assets came from verified or user-authorized sources
- Product names, feature wording, claims, metrics, slogans, and CTA match official sources or are clearly marked as concept copy
- The video duration, aspect ratio, and language match the brief
- Copy is readable and not overcrowded
- The final logo is not stretched, cropped, or rebuilt
- Motion has clear visual ownership and does not obscure the product
- The output is on the canvas and multi-asset outputs are grouped

If the output fails an authenticity check, replace the questionable asset with an official/user-authorized source or stop and ask for the asset. Never improve an imitation.

## STEP 10: Deliver

Provide:

- Final video path or canvas output
- Duration, aspect ratio, and language
- Short creative summary
- Provenance manifest or source summary
- Rights/disclaimer note when needed
- Specific suggestions for the next iteration, such as pacing, claim clarity, CTA, audio, or platform crop

## Failure recovery

- Wrong or approximate logo: remove it, locate the current official file or ask for the user's original, then regenerate or re-edit.
- Fake-looking product/UI: replace with official, user-supplied, or licensed media. Do not polish the imitation.
- Beautiful but generic: add a complete product interaction, verified claim, or real output.
- Fast but chaotic: reduce simultaneous actions, assign a visual owner, and preserve matched motion across cuts.
- Smooth but slow: shorten holds, overlap transitions, and brake only around key messages.
- Asset unavailable: ask for an authorized original; never guess.

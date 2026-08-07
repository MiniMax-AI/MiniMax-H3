# Full-Reference Mode Rewrite Output Format Guide

This guide explains how rewrite outputs are organized and written in full-reference mode.

Write all six rewrite sections in English. Preserve the original language only for dialogue and lyrics inside `<d>` and for text visibly present in the scene.

**Description detail:** Make `detailed_description` as detailed and explicit as possible. For each shot, clearly establish the current composition, subject appearance and position, environment and lighting, actions and state changes, camera movement, current sound, and the points where referenced content actually appears or takes effect. Avoid reducing the description to a plot summary or a list of reference relationships.

> The basic formats for shots, camera movement, speakers, dialogue, and ordinary sound are shared with the Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA). This guide focuses on the reference labels, analysis sections, and format differences specific to full-reference mode.

## 1. Overall Structure

A complete rewrite output consists of six sections in the following order:

| Section | Purpose |
| --- | --- |
| `subject_definitions` | Defines referenced content and its reference labels |
| `summary` | Summarizes the task type, target video, and main reference relationships |
| `retention_analysis` | Describes how referenced content is preserved, transferred, or reused |
| `detailed_description` | Describes visuals, actions, shots, sound, and dialogue in playback order |
| `overall_soundscape` | Summarizes ambience and physical sounds |
| `non_diegetic_music` | Describes background music audible only to the audience |

## 2. Reference Labels and Definitions (`subject_definitions`)

Full-reference rewrites use four types of labels to identify the source and role of referenced content:

| Label | Meaning |
| --- | --- |
| `<Subject N>` | Visible content abstracted from reference assets that can be reused or modified in the target video |
| `<Picture N>` | A reference image used as a concrete target frame or shot-planning anchor |
| `<Video N>` | A reference video that provides an editing source, continuation starting point, or whole-video temporal structure |
| `<Audio N>` | An audio signal that is copied or referenced |

> Once a reference label is assigned to a piece of content, it keeps the same meaning across `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, and the audio sections.

`subject_definitions` defines each piece of referenced content that must be tracked separately later, such as a person, an environment, a source video's structure, or an audio track. Give each item its own line and explain what its label denotes, its reference role, and the main features to follow; name the corresponding source asset when its provenance needs to be made explicit. If `<Picture N>` or `<Video N>` only identifies the source of another referenced item and will not be analyzed or used separately later, cite it inside that item's definition without adding a separate line. `retention_analysis` records where each referenced item appears and whether it is fully preserved, partially preserved, transferred, or reused.

### 2.1 `<Subject N>`

`<Subject N>` is used for reusable visible content, including:

- People, animals, or objects
- Scenes, backgrounds, or environments
- Clothing, props, interfaces, or visual effects
- Styles, actions, expressions, or poses

It represents a content unit that will actually be used in the target video, rather than the source file itself. One subject may be defined by multiple reference assets, and one reference asset may provide multiple subjects.

```text
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.
```

When the same subject comes from multiple assets, combine the sources and state what each asset provides:

```text
<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.
```

### 2.2 `<Picture N>`

Use a standalone `<Picture N>` when the reference image itself serves as a shot's first frame, keyframe, last frame, edited keyframe, or composition anchor:

```text
<Picture 2> is the first frame of [Shot 1], showing a woman seated beside a café window.
```

If an image is used only to define a character, scene, costume, or style, do not create a standalone picture entry. Instead, cite the image source inside the corresponding `<Subject N>` definition.

When an image acts as a storyboard or shot-planning reference, state which shots it maps to and what planning information it provides:

```text
<Picture 3> is a storyboard reference for [Shot 1] and [Shot 2], defining their viewpoint, subject placement, and shot order.
```

### 2.3 `<Video N>`

`<Video N>` is reserved for whole-video relationships, such as:

- Editing an original video
- Continuing from the end of an original video
- Referencing the original video's camera movement, cuts, rhythm, or temporal structure

```text
<Video 1> is the source video for the target video edit.
```

If a person, object, scene, action, or effect from a reference video is reused as visible content, it still belongs under `<Subject N>`. `<Video N>` identifies the asset or structural source and does not replace subject labels.

### 2.4 `<Audio N>`

`<Audio N>` represents a standalone audio asset or an enabled synchronized audio track from a reference video. Common uses include:

- Copying all or part of an audio signal
- Referencing a background-music style
- Referencing a speaker's voice timbre and delivery
- Using dialogue, lyrics, or sound effects from the original audio
- Referencing beat, rhythm, or audio continuity

When an `<Audio N>` explicitly corresponds to a target speaker, reuse that speaker's global ID in the definition: write `<Subject N> (Sx)` when the speaker maps to a defined subject, or use a stable voice description followed by `(Sx)` otherwise. The ID comes from the target video's global speaker order and is not independently assigned or renumbered in the audio definition. See Section 5.4 for the speaker-numbering rules:

```text
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
```

When one audio asset serves multiple roles, describe those roles in one natural sentence rather than creating additional subsections.

### 2.5 Visual and Audio Tracks from the Same Reference Video

`<Video N>` and `<Audio N>` are numbered independently. Each index indicates only the label's order within its own category and does not encode a pairing between the two categories. The same reference video may therefore correspond to `<Video 1>` and `<Audio 2>`; different indices do not prevent them from coming from the same source asset.

An ordinary reference video does not create `<Audio N>` merely because the file contains sound.

An `<Audio N>` definition primarily states the audio's role and does not have to name the `<Video N>` it comes from. State the shared source only when needed to remove provenance ambiguity, for example:

```text
<Video 1> is the source video for the target video edit.
<Audio 2> is the synchronized audio track of <Video 1> and is reused in the target video.
```

## 3. `summary`

This section uses one short English paragraph to summarize the target video and its reference relationships. It begins with a square-bracketed task-type prefix:

```text
[reference generation] ...
[video editing + reference generation + audio reuse] ...
```

Choose task types according to the actual role each reference asset plays in the target video:

| Task type | When to use it |
| --- | --- |
| `keyframe completion` | An image serves as the target video's first frame, keyframe, last frame, edited keyframe, or another concrete frame anchor |
| `reference generation` | An image, video, or audio asset provides generation guidance for a character, scene, style, action, camera movement, storyboard, and so on, without serving as a concrete frame or as the source video being edited or continued |
| `video editing` | An existing source video is directly modified; editing an image or generating between still keyframes does not belong to this type |
| `video continuation` | New content continues, extends, resumes, or transitions from an existing source video |
| `audio reuse` | The same audio signal is reused in full or in part |
| `audio reference` | The audio signal is not copied directly; only its music style, timbre, dialogue or lyric content, sound-effect texture, beat, or continuity is referenced |

When a task satisfies multiple relationships, combine the task types with ` + ` and do not repeat a type. For example, continuing from a source video while using an image as the last frame is written as `[video continuation + keyframe completion]`. Editing a source video while retaining its original audio may be written as `[video editing + audio reuse]`.

The mere presence of video or audio does not automatically create a corresponding task type. If a reference video provides only camera movement, cuts, or rhythm, it normally belongs to `reference generation`. Use `video editing` or `video continuation` only when that video is directly edited or continued.

When editing a source video, use `audio reuse` as well if its original audio remains audible. When continuing a source video without directly copying the audio signal, use `audio reference` if the new audio only continues the original track's audible characteristics.

The summary uses the previously defined `<Subject N>`, `<Picture N>`, `<Video N>`, and `<Audio N>` labels to describe the main subjects, shot flow, and roles of the reference assets. Do not introduce new reference labels in this section.

For video-editing tasks, begin the summary after the task-type prefix with:

```text
The target video is an edited version of <Video 1>.
```

## 4. `retention_analysis`

This section describes how each piece of referenced content is preserved, transferred, copied, or referenced in the target video. Use one line for each reference label and preserve the meaning established in `subject_definitions`.

### 4.1 Visible Content

`<Subject N>`, `<Picture N>`, and `<Video N>` use the following relationship markers. These markers are fixed English values in the output format:

| Relationship marker | Meaning |
| --- | --- |
| `fully_preserved` | The defined role of the referenced content is fully preserved |
| `partially_preserved` | The referenced content is still used, but some defined characteristics are changed or only partially retained |
| `attribute_transfer` | Referenced characteristics are transferred to a different identifiable target subject |
| `weak_reference` | Only broad similarity in style, category, composition, or atmosphere is retained |

Subject entry:

```text
<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
```

Picture entry:

```text
<Picture 2> ([Shot 1] first frame): fully_preserved - ...
```

Video-structure entry:

```text
<Video 1> (cut and pacing structure): weak_reference - ...
```

### 4.2 Audio

`<Audio N>` uses the following relationship markers:

| Relationship marker | Meaning |
| --- | --- |
| `fully_copy` | The complete source audio serves as the target video's complete final audio track |
| `partially_copy` | Only part of the timeline or selected audio layers are copied, or other sounds are added, removed, or replaced after copying |
| `reference` | The signal is not copied directly; only timbre, rhythm, music style, dialogue content, or sound texture is referenced |
| `weak_reference` | Only broad similarity in category or atmosphere is retained |

```text
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.
```

```text
<Audio 2>: reference - the target speaker follows <Audio 2>'s voice timbre and measured delivery without copying the original signal.
```

Choose each relationship marker only within the reference role already defined for that label in `subject_definitions`. Do not treat newly added actions, backgrounds, or plot events in the target video as losses of reference fidelity.

## 5. `detailed_description`

This is the main body of a full-reference rewrite. It describes visuals, actions, sound, and dialogue shot by shot in target-video playback order and inserts reference labels where they apply.

### 5.1 Basic Format

The basic format follows the Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA):

- Write the body in English. Preserve the original language of dialogue, lyrics, and visible text.
- `[Shot 1]` marks the opening shot and has no timestamp. Later shots use `[Shot N] At MM:SS.mmm, ...` to mark cut times.
- Write camera movement as natural English within the current shot, including movement type, amplitude, and speed when they need to be expressed.
- Give vocal sources stable `(S1)`, `(S2)`, and subsequent IDs. Write dialogue and lyrics as `<d>[Language] ...</d>`.
- Use `<scenetrans>`, `<cutoff>`, and the corresponding continuity descriptions for dialogue crossing a cut, speech truncated by the video ending, and continuous audio across shots.

For complete rules and examples covering camera vocabulary, group speech, voice-over, dialogue across cuts, and visible text, see the Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA).

### 5.2 Full-Reference Mode Differences

| Dimension | T2VA | Full-reference mode |
| --- | --- | --- |
| Main field | `integrated_multimodal_description` | `detailed_description` |
| Style opening | Written after `[Shot 1]` | Established in one or two English sentences before `[Shot 1]` |
| Reference information | Does not use full-reference labels | Inserts `<Subject N>`, `<Picture N>`, `<Video N>`, and `<Audio N>` at their first appearance and where their roles apply |
| Audio relationships | Describes the target video's own sound | Cites `<Audio N>` in the corresponding shot or audio phase and states whether the signal is copied or referenced |

Opening example:

```text
The target video is in a cinematic, literary music-video style with soft lighting and a slightly desaturated color palette.
[Shot 1] The scene opens in a crowded urban street...
[Shot 2] At 00:09.000, the shot cuts to an extreme close-up...
```

For generation tasks, `detailed_description` is normally 350-500 English words. Dialogue-dense content prioritizes fitting the complete spoken timeline rather than mechanically reaching a word count. Video-editing descriptions scale with the complexity of the source video and do not have to follow the generation-task range. A single shot does not automatically justify a shorter description; distribute detail across multiple shots according to their information load.

### 5.3 Using Reference Labels in Shots

At the first clear appearance of an important `<Subject N>`, describe its referenced characteristics, position in the frame, and current action within what is actually visible in the shot. Continue using the same label in later shots without redefining what the label represents.

Use natural phrasing for concrete frame anchors:

```text
the shot begins from <Picture 1>
the shot's keyframe corresponds to <Picture 2>
the shot ends on <Picture 3>
```

When editing or continuing an original video, cite `<Video N>` naturally where its source state, structure, or continuation relationship applies. Cite `<Audio N>` in the shot or semantic phase where the audio relationship is active.

### 5.4 Speakers, Audio Sources, and Dialogue

The basic speaker-ID and `<d>` formats follow T2VA. When a referenced subject physically speaks, retain both the visual reference label and the speaker ID:

```text
<Subject 2> (S1) turns toward the woman and says, <d>[English] Last summer, I went to my grandfather's house. He talked about you.</d>
```

`<Subject N>` identifies the referenced subject, while `(Sx)` identifies the actual speaker. When the subject speaks, write `<Subject N> (Sx)`. If the same subject speaks off-screen, keep the same form and mark it as `off-screen`. When the speaker does not correspond to a defined subject, use a stable voice description followed by `(Sx)`.

When verbal content is only a cue within a directly reused BGM or complete soundtrack, and no person, character, narrator, or other independent vocal source physically produces it, use `<Audio N>` as the audible source and do not invent an additional `(Sx)`. If a concrete person, character, narrator, or other independent vocal source produces the voice, assign and reuse `(Sx)` for that source:

```text
When <Audio 1> reaches the phrase <d>[English] I'm lonely lonely lonely lonely lonely I'm lonely</d>, <Subject 1> performs the corresponding hand gesture without becoming a separate speaker source.
```

When dialogue, narration, or lyrics from reference audio are directly reused, or when the input prompt explicitly requests their reperformance, preserve the exact source words and original language inside `<d>`. Write `[unclear]` for unintelligible spans instead of guessing or paraphrasing them. Standardize punctuation to the basic written marks needed to express the sentence, such as `,`, `.`, `?`, and `!`; remove repeated tildes, emoji, bullets, and repeated or decorative punctuation. End complete statements, questions, and exclamations with `.`, `?`, or `!` respectively before `</d>`.

When only timbre, rhythm, emotion, or delivery is referenced, do not carry the original dialogue from the reference audio into the target video.

Assign `(Sx)` once according to the order of actual vocal events in the target video. Reuse the corresponding ID at every actual vocal event in `detailed_description`; an `<Audio N>` definition bound to a target speaker in `subject_definitions` also reuses the same `(Sx)` but never assigns a new one independently. Do not write `(Sx)` in `retention_analysis`. Verbal cues that exist only within a directly reused BGM or complete soundtrack use `<Audio N>`; voices physically produced by a concrete person, character, narrator, or other independent vocal source use `(Sx)`.

## 6. `overall_soundscape` and `non_diegetic_music`

The definitions of these two sound categories follow the Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA).

`overall_soundscape` summarizes ambience and physical sounds across the full video. Dialogue, singing, and sound events synchronized to a particular shot remain in `detailed_description`:

```text
overall_soundscape: Quiet indoor room tone and a low ventilation hum continue throughout the video.
```

`non_diegetic_music` describes background music that the characters cannot hear and that is audible only to the audience. When music is present, state its instrumentation, tempo, and dynamic development:

```text
non_diegetic_music: A restrained solo-piano score at a slow tempo, with sustained low cello underneath and no swell.
```

When reference audio is used, state its copy or reference relationship only in the section that matches the audible layer: ambience and sound effects belong in `overall_soundscape`, while audience-only score belongs in `non_diegetic_music`. If the same audio provides both kinds of content, describe the corresponding relationship in each section:

```text
overall_soundscape: The copied ambience layer from <Audio 1> continues throughout the target video.
non_diegetic_music: <Audio 2> is directly reused as the complete audience-only score.
```

Write complete dialogue and lyrics only inside `<d>` in `detailed_description`; do not repeat them in these two sections.

## 7. Complete Example

<details>
<summary>Show the complete example</summary>

```text
subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A
```

</details>

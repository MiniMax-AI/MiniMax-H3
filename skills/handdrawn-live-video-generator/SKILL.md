---
name: handdrawn-live-video-generator
description: |
  For creators making surreal short videos that blend rough glowing hand-drawn animation with live-action spaces. Users provide a scene idea, contact object or hand, desired mood, and optional language or style constraints. The Skill clarifies the physical contact, designs continuous morphing, escape route, and delayed handheld chase movement, then writes a reusable 15-second 16:9 video prompt in the user's language. After user confirmation it recommends MiniMax H3 generation and checks contact realism, camera delay, rough glowing stroke texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene cuts.
trigger-words: [手绘发光动画实拍融合, 15秒变形追逐视频提示词, Seedance视频prompt, H3生成视频, 视频prompt, 手绘动画接触真实物体, 蜡笔粉笔质感, 多语言视频提示词]
---

# Handdrawn Live-Action Fusion Video Generator

Use this Skill when the user wants a **finished 15-second, 16:9 live-action-and-hand-drawn fusion video**. The output must preserve the structure: a flat hand-drawn luminous animation appears in a real space, clearly contacts live-action hands or objects in the first 0-3 seconds, continuously morphs as one single entity, escapes, and a handheld phone camera follows slightly late.

This Skill **organizes the prompt in the user input language and recommends MiniMax H3 as the confirmed generation step**. Do not generate video until the user explicitly confirms H3 generation. Do not route to planner or executor for prompt-writing. The final prompt language must follow the dominant language of the user input: Chinese input produces Chinese, English input produces English, Japanese input produces Japanese; for mixed input, use the dominant language; when unclear, use the current conversation language. Only user-required proper nouns, model names, or literal parameters may remain unchanged.

## Step 1: Understand the user intent as same-language constraints

When the user provides any language, understand it as the following workflow requirements and express the final prompt in the dominant language of the user input:

- 基于参考 prompt 创作一个全新的 15 秒视频生成 prompt，不要表面模仿原句。
- 必须保留影像结构：实拍空间中出现平面的手绘发光动画；动画与真实手或真实物体接触；同一个存在连续变形并逃跑；相机总是慢半拍追赶。
- 手绘动画质感必须像蜡笔、粉笔、彩色铅笔、粉彩、粗糙笔刷；线条轻微抖动，有涂抹不均、毛边和逐帧重画感。
- 禁止 3DCG、毛绒玩具感、均匀矢量线、平滑霓虹、恐怖怪物、巨大眼睛、裂口、牙齿、威吓、扑咬、突然黑屏、跳吓。
- 0-3 秒必须出现实拍手与手绘动画的清晰接触，例如缠住手指、落在掌心、被抓时逃跑、从指尖诞生。
- 动画必须作为同一个实体连续变形，可在“线条、生物、记号、植物、交通工具、生活小物”等形态之间变化，并保留前一形态的痕迹。
- 不允许突然出现另一个全新角色。
- 全片在同一空间或相邻范围内连续展开，不用剪辑跳到另一个地点，像拍摄者真的边走边追。
- 每个区间 0-3、3-6、6-10、10-13、13-15 秒都要有新的变形、移动、接触、发现、恶作剧或惊喜。
- 拍摄者也必须参与：伸手、抓、追、打开门或盒子、接住、后退、被恶作剧等。
- 调性是可爱、生活感、怀旧、温柔、略带切感，不是恐怖喜剧。
- 13-15 秒必须有空间级变形：之前的线扩散到墙、地板、天花板、窗户、水槽或通道，变成巨大花、星空、夕阳、云、丝带、涂鸦小镇等；结尾要有感动余韵和一点可爱笑点。

## Step 2: Invent all creative content fresh

For every run, create a new combination. Do not reuse the reference prompt's dark room, PC, fridge, stars, hearts, blue vortex, butterfly, snake, octopus, hamburger, giant eye, teeth, or dark ending.

Choose new values for all of these:

1. 实拍空间，但仍是生活化且可连续追踪的相邻范围；
2. 手绘实体或初始形态；
3. 中心色彩；
4. 连续变形链路；
5. 与真实手或实物的接触方式；
6. 在空间中的追逐路线；
7. 拍摄者反应；
8. 最后的空间级变形与可爱笑点。

Good example directions include: 雨天厨房水槽、旧阳台晾衣角、清晨玄关、小书店走廊、火车窗边小桌、浴室镜柜、手作桌、温室通道、自助洗衣店长椅、旧餐桌. Use only one or adjacent connected areas.

## Step 3: Required output format

The final prompt must begin with a sentence pattern matching the dominant language of the user input. For Chinese input, use this pattern:

`15秒，16:9横版视频。将实拍的〇〇与手绘发光动画融合的影像。`

Replace “〇〇” with the newly invented live-action space or everyday scene. For non-Chinese input, use an equivalent opening in the same language and include duration, 16:9 landscape format, live-action space, and hand-drawn glowing animation fusion.

Do not add titles, auxiliary headings such as 舞台设置 or 色味, or explanatory notes. Keep this paragraph order:

1. 开头句；
2. 实拍空间与手机拍摄质感；
3. 0-3 秒；
4. 3-6 秒；
5. 6-10 秒；
6. 10-13 秒；
7. 13-15 秒；
8. 手绘质感；
9. 相机追随方式；
10. 禁止事项；
11. 环境音。

## Step 4: Prompt writing rules

- Keep the prompt executable as a video-generation prompt, not an essay.
- Unless the user asks for explanation, the final answer should contain the prompt text in the dominant language of the user input first, followed by one short next-step recommendation in the same language.
- The recommendation must invite the user, in the same language, to use MiniMax H3 to generate a 15-second 16:9 video from this prompt. Chinese example: `下一步建议：如果你确认这个 prompt，我可以继续用 H3 模型生成 15 秒 16:9 视频。`
- Do not generate a video, image, audio, storyboard, or intermediate asset until the user explicitly confirms the H3 generation step.
- Mention Seedance only as a target usage context when the user asks; do not add model parameters inside the creative prompt unless requested.
- The first 0-3 seconds must make real/hand-drawn fusion obvious through contact.
- The camera must not neatly center the animation. It should lag behind, pan/tilt/advance after the entity already leaves the frame edge.
- The entity should remain traceable: each new form preserves a line, tail, color smear, body curve, or motif from the previous form.
- Use soft, emotional, comic, cute beats: tiny stumbles, shy gestures, a dot falling behind, a petal sticking to lens, a star dropping into a palm, confetti sneeze, one small creature lagging behind.
- Avoid any horror-coded anatomy or threatening motion.
- The final prompt must not randomly mix in vocabulary, transition sentences, or writing systems from outside the user input language; when the user input language is Japanese, natural Japanese is allowed.

## Step 5: Delivery

Output the final prompt in the dominant language of the user input first. After the prompt, add exactly one short same-language recommendation line inviting the user to continue with MiniMax H3 video generation. Do not include a title, checklist, model call summary, filename, or canvas delivery note. If the user later confirms H3 generation, generate the 15-second 16:9 video with MiniMax H3 and check that the result preserves contact, continuous morphing, delayed camera chase, and non-horror hand-drawn texture.

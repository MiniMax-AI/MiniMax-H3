<div align="center">
  <img width="100%" src="assets/minimax-h3-header.gif" alt="MiniMax H3">
</div>

<p align="center">
  <a href="https://hailuoai.video" target="_blank"><img src="https://img.shields.io/badge/Hailuo%20AI-FF6C37?logo=minimax&logoColor=white" alt="Hailuo AI"></a>
  <a href="https://platform.minimax.io/docs/guides/text-generation" target="_blank"><img src="https://img.shields.io/badge/API-FF6C37?logo=minimax&logoColor=white" alt="API"></a>
  <a href="https://www.minimax.io" target="_blank"><img src="https://img.shields.io/badge/MiniMax%20Website-FF6C37?logo=minimax&logoColor=white" alt="MiniMax Website"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-H3" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <br>
  <a href="https://modelscope.cn/organization/minimax" target="_blank" rel="noopener noreferrer"><img alt="ModelScope MiniMax AI" src="https://img.shields.io/badge/ModelScope-MiniMax%20AI-white?labelColor=%23EF3D5D"></a>
  <a href="https://platform.minimaxi.com/docs/faq/contact-us" target="_blank"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="https://discord.com/invite/dbMxutw7tP" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-4CAF50?logo=creativecommons&logoColor=white" alt="LICENSE"></a>
</p>

# MiniMax H3

## Prompt Writing Skill

Install the H3 prompt writing skill — one of nine skills bundled with this repository:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

It ships with two prompt guides under `skills/h3-prompt-writing/references/`: `base-en.txt` for text/keyframe modes and `ref-en.txt` for full-reference (Ref2VA) mode. The remaining eight are style-specific video generation skills:

<table align="center">
  <tr>
    <td align="center"><img src="assets/minimalist-product-ad-generator.gif" alt="minimalist-product-ad-generator" width="240"><br><a href="skills/minimalist-product-ad-generator/SKILL.md">minimalist-product-ad-generator</a></td>
    <td align="center"><img src="assets/3d-animation-short-generator.gif" alt="3d-animation-short-generator" width="240"><br><a href="skills/3d-animation-short-generator/SKILL.md">3d-animation-short-generator</a></td>
    <td align="center"><img src="assets/papercraft-stop-motion-explainer.gif" alt="papercraft-stop-motion-explainer" width="240"><br><a href="skills/papercraft-stop-motion-explainer/SKILL.md">papercraft-stop-motion-explainer</a></td>
    <td align="center"><img src="assets/brand-promo-video-generator.gif" alt="brand-promo-video-generator" width="240"><br><a href="skills/brand-promo-video-generator/SKILL.md">brand-promo-video-generator</a></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/music-video-subtitle-generator.gif" alt="music-video-subtitle-generator" width="240"><br><a href="skills/mv-subtitle-skill-confirmed/SKILL.md">music-video-subtitle-generator</a></td>
    <td align="center"><img src="assets/co-op-game-intro-generator.gif" alt="co-op-game-intro-generator" width="240"><br><a href="skills/co-op-game-intro-generator/SKILL.md">co-op-game-intro-generator</a></td>
    <td align="center"><img src="assets/paper-collage-explainer-generator.gif" alt="paper-collage-explainer-generator" width="240"><br><a href="skills/paper-collage-explainer-generator/SKILL.md">paper-collage-explainer-generator</a></td>
    <td align="center"><img src="assets/handdrawn-live-video-generator.gif" alt="handdrawn-live-video-generator" width="240"><br><a href="skills/handdrawn-live-video-generator/SKILL.md">handdrawn-live-video-generator</a></td>
  </tr>
</table>

## Online API
Use MiniMax\-H3 directly via API\. 
- Global: [platform\.minimax\.io](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) \| CN: [platform\.minimaxi\.com](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create)

## Online App
Use MiniMax\-H3 directly via App\.
- WebApp Global: [hailuoai\.video](https://hailuoai.video/tools/minimax-h3) \| CN: [hailuoai\.com](https://hailuoai.com/)
- Desktop Global: [hub\.minimax\.io](https://hub.minimax.io/) \| CN: [hub\.minimaxi\.com](https://hub.minimaxi.com/)


## System Overview
MiniMax H3 is a general-purpose, omni-modal generative system. It supports unified understanding of multimodal contexts composed of text, images, video, and audio, and can generate video with native stereo audio at resolutions up to 2K and durations of up to 15 seconds. Thanks to its task-generalization-oriented system design, H3 already possesses broad multimodal context understanding and generation capabilities at the pre-training stage, enabling outstanding performance in following complex multimodal instructions.

H3 supports the following input and output specifications:

| Category | Specification |
|---|---|
| Output duration | 4–15 seconds |
| Output aspect ratio | Supports a wide range of aspect ratios, including but not limited to 21:9, 16:9, 4:3, 1:1, 3:4, and 9:16 |
| Output resolution | Supports various resolution dimensions. The shorter side is set to 768 pixels by default. 2K \| generation can be achieved with H3-Regenerate-2K |
| Output frame rate | 24 FPS |
| Output audio | 32 kHz stereo |
| Supported dialogue languages | Stable support for 11 languages: Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Additional languages are also supported to varying degrees |

### Model Variants and Input Specifications

| Model Variant | Input Mode | Specifications |
|---|---|---|
| H3-Base-FL2VA | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No image input: Text-to-video mode <br>- One image input: First-frame-to-video or last-frame-to-video generation <br>- Two image inputs: First-and-last-frame-to-video generation |
| H3-Base-Ref2VA | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:** ≤ 9 images <br>- **Videos:** ≤ 3 clips; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Audio:** ≤ 3 clips; audio must be accompanied by image or video input and cannot be used as the sole input; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Mixed inputs:** Maximum number of files across all input types is 12 |

![Image](assets/overview.png)

The complete H3 system consists of the following three modules:
- H3-Context-IR: As inputs become increasingly complex, we build a dedicated system to deeply understand and refine the input multimodal instructions, then convert them into a form that H3 can readily understand—the Context Intermediate Representation—for generation. **H3-Context-IR is critical to the quality of the final output, so we strongly recommend incorporating it into your generation pipeline or following the “Prompting Guidance” to build your own context-processing system.**
- H3-Base: Generates audio and video based on the H3-Context-IR output, producing results at 768p resolution.
- H3-Regenerate-2K: Feeds the 768p result together with the original context back into H3 to regenerate the output at 2K resolution. This process leverages both H3’s powerful generative capabilities and the rich information contained in the original context, enabling it to produce high-resolution outputs with more accurate details and greater visual fidelity.

## Model Architecture

### H3\-Context\-IR

H3\-Context\-IR is a hosted preprocessing and orchestration system designed for free\-form multimodal inputs\.

It interprets the relationships among text, images, audio, and reference videos, as well as how these materials relate to the intended generation output\. Its internal workflow includes instruction parsing, cross\-modal association, temporal understanding, and complex logical reasoning\.

H3\-Context\-IR serializes its understanding of the context into a structured representation accepted by H3\-Base\. Without deviating from the user’s original intent, it may also supplement missing or underspecified semantic details where appropriate\.

Because H3\-Context\-IR relies on a multi\-stage workflow and multiple hosted models and services, it is not included in this open\-source release\. We provide an API that enables users to reproduce the behavior of the official workflow\. We also provide detailed tutorials, and developers can follow the **Prompting Guidance** to build their own preprocessing systems\.

For detailed usage instructions, see **Recommended Workflow — Full 2K Workflow**\.

**Safety Guardrails**

User\-submitted text, images and videos, as well as enhanced prompts, are subject to automated moderation\. Content suspected of being unlawful, pornographic, or infringing third\-party rights may be blocked\. We use industry\-standard filtering measures but cannot eliminate false positives or false negatives\. These guardrails do not affect the Licensee’s obligations under the MiniMax H3 Community License, especially those relating to lawful use and use restrictions\.

### H3\-Base

![Image](assets/full-arch.png)

#### Architecture Overview

- H3\-Base encodes different modalities using their corresponding encoders or VAEs and organizes the encoded representations into a unified packed multimodal sequence\. RoPE is used to capture the necessary spatial and temporal relationships among tokens before the entire sequence is passed to the H3\-Omni\-Transformer\.

- Specifically, text is encoded by the H3\-Encoder; visual inputs are encoded by both the H3\-Encoder and the H3\-VisualVAE; and audio is encoded solely by the H3\-AudioVAE\.

- The H3\-Omni\-Transformer jointly predicts video and audio latents, which are then decoded into video and stereo audio, respectively\.

- To reduce the computational cost of long multimodal sequences, H3 natively supports sparse\-attention training and inference\. The initial open\-source release provides inference with full attention only\. Our sparse\-attention implementation will be released in a future update\.

#### H3\-Encoder

- The H3\-Encoder uses the full pretrained weights of Qwen3\-VL\-32B and provides the hidden states from its 50th layer to the H3\-Omni\-Transformer\.

- We add several special tokens, such as `<d>`, to the tokenizer configuration\. When using H3, the tokenizer and associated configuration files provided in the H3 repository are required\.

#### H3\-VAE

H3 uses separate visual and audio latents to represent their respective modalities\.

##### H3\-VisualVAE

- H3\-VisualVAE is a temporally causal video autoencoder with a spatial compression factor of 16×, a temporal compression factor of 4×, and 24 latent channels, denoted as f16t4d24\. We apply several latent\-space optimization techniques to jointly improve reconstruction quality and latent learnability\.

- Before being passed to the H3\-Omni\-Transformer, the visual latents are further patchified with a patch size of `1 × 2 × 2` along the `(time, height, width)` dimensions\. As a result, the visual tokens entering the Transformer have an effective spatial downsampling factor of 32×, while the temporal downsampling factor remains 4×\.

- The latent space of H3\-VisualVAE is optimized for both reconstruction quality and ease of learning by the generative model\. After training its encoder, we additionally train a ViT\-based decoder to reduce decoding costs and further improve reconstruction quality\.

##### H3\-AudioVAE

- H3-AudioVAE uses the same encoder and decoder for both the left and right audio channels while processing each channel independently. The decoded channels are then recombined, enabling stereo audio input and output.
- For each channel, H3-AudioVAE compresses 32 kHz audio into a sequence of latent tokens with a temporal rate of 40 Hz.
- Inspired by VA-VAE, we optimize the latent space to preserve audio reconstruction quality while making it easier for the generative model to learn.

#### H3\-Omni\-Transformer

- For scalability and generalization, we adopt a relatively simple Transformer block design\. H3\-Omni\-Transformer is a 33B\-parameter dense, single\-stream Transformer, with approximately 13B parameters residing in AdaLN\-related branches\. Because the AdaLN modulation outputs can be precomputed and cached, these parameters do not need to be loaded for inference\-only deployment\. We release the complete model weights to support further development, including fine\-tuning\.

- Neither the attention layers nor the FFN layers contain modality\-specific structures\. Modality\-specific parameters are confined to the input/output layers and the AdaLN branches\. In particular, modality\-specific AdaLN improves generation quality with relatively low additional training and inference costs\.

- The model uses three\-dimensional Multimodal Rotary Position Embeddings \(MM\-RoPE\) to represent positional relationships across the temporal and two spatial dimensions, `(t, h, w)`\.

- During the final stage of training, we introduce native sparse attention to reduce the computational cost of long sequences\. The sparse\-attention implementation is not included in the initial open\-source release and will be published separately in a future update\.

    

### H3-Regenerate-2K

- For H3's 2K\-resolution output, instead of using a conventional dedicated super\-resolution module, we use the H3 base model to regenerate its own low\-resolution result through an in\-context manner\.

- This approach provides two advantages: \(1\) the regeneration process can reuse the generative capabilities of H3 base model to the greatest extent possible; and \(2\) the in\-context format can reuse the original multimodal context when producing high\-resolution output, allowing it to recover information that conventional super\-resolution methods would otherwise have to “guess,” such as small text and fine details\.

- In\-context regeneration is also an example of task generalization\.

- **Due to the complexity of the system, this module is not yet open\-sourced\. We will release it once it is ready\.** We provide an API for validating the official results; see "Full 2K Workflow" below\.



## Recommended Workflow

To help the community deploy MiniMax H3 correctly, we provide two validation methods\.

Since the complete H3 system consists of three modules—H3\-Context\-IR, H3\-Base, and H3\-Regenerate\-2K—the “Full 2K Workflow” provides an end\-to\-end validation pipeline for 2K output, combining the Open Platform API with a locally deployed H3\-Base\. The “Local Deployment of H3\-Base” section provides a method for validating 768p output using only a locally deployed H3\-Base\.

In addition, the “Prompting Guidance” section provides a detailed tutorial to help the community develop their own prompting systems\.

### Local Deployment of H3\-Base

MiniMax H3 is released as two task\-specific checkpoints\. Each checkpoint contains a specialized Omni Transformer Model together with the required processor, tokenizer, text encoder, Visual VAE, and standalone Audio VAE components\.

|Checkpoint|Supported Tasks|Input Conditions|Output|Precision|
|---|---|---|---|---|
|MiniMax\-H3 Base FL2VA|Text\-to\-Audio\-Video \(`t2va`\), First/Last\-Frame\-to\-Audio\-Video \(`fl2va`\)|Text; optional first frame, last frame, or both|Video and audio|BF16|
|MiniMax\-H3 Base Ref2VA|Reference\-to\-Audio\-Video \(`ref2va`\)|Text with reference images, videos, and/or audio|Video and audio|BF16|

The released checkpoints are CFG-distilled Omni Transformer model weights. 

Each checkpoint is distributed as a self\-contained Hugging Face\-style repository with the following components:

```text
<TASK>/
├── model_index.json
├── processor/
├── tokenizer/
├── text_encoder/
├── transformer/
├── visual_vae/
└── audio_vae/
```

Download the model. The repository hosts the original checkpoint (`FL2VA/`, `Ref2VA/`) and the diffusers format side by side, so scope the download to what your framework needs:

`model_index.json` is the repository-level public entry. The task-family-specific diffusers indexes remain under `FL2VA/model_index.json` and `Ref2VA/model_index.json`.

```bash
# Original checkpoint, both task families (SGLang, vLLM):
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3

# Or a single task family:
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

diffusers users do not need a manual download: `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` fetches exactly the components it needs. See the [diffusers documentation](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md) for loading recipes.

We recommend the following inference frameworks to serve the model:

- [SGLang](https://docs.sglang.io/) \- see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) 

- [vLLM](https://github.com/vllm-project/vllm) \- see [vllm recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)

- [diffusers](https://github.com/huggingface/diffusers) \- see [diffusers docs](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md)

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) \- see  [Comfy tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3); use [R2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) / [T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)

#### Sglang Deployment

Here we use sglang as a deployment example\. See the [MiniMax\-H3 deployment guide](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#3-serve-minimax-h3) for additional deployment configurations\.

FL2VA:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant fl2va
```

Ref2VA:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30011 \
  --model-variant ref2va
```

#### Reproducible 768p cases

The following three use cases T2VA, FL2VA, and Ref2VA demonstrate how to reproduce MiniMax\-H3 video\-audio generation\.

| Use case | Request | Result |
|---|---|---|
| T2VA | [View script](scripts/readme/reproducible-768p-t2va-request.sh) | [t2va.mp4](assets/t2va.mp4) |
| FL2VA | [View script](scripts/readme/reproducible-768p-fl2va-request.sh) | [fl2va.mp4](assets/fl2va.mp4) |
| Ref2VA | [View script](scripts/readme/reproducible-768p-ref2va-request.sh) | [ref2va.mp4](assets/ref2va.mp4) |

### Full 2K\-Workflow

This section explains how to combine a locally deployed SGLang service with the official **H3\-Context\-IR** and **H3\-Regenerate\-2K** APIs to reproduce the quality of 2K videos generated directly by the MiniMax API\.
Before you begin, configure the SGLang endpoint and your MiniMax API credentials:

```bash
# URL of your SGLang deployment
SGLANG_DEPLOYMENT_URL="<sglang-deployment-url>"

# MiniMax API endpoint (choose one)
# CN
MINIMAX_API_BASE="https://api.minimaxi.com"
# Global
# MINIMAX_API_BASE="https://api.minimax.io"

# API token obtained from the MiniMax platform
TOKEN="<token>"
```

MiniMax platform:

API docs:
- Create H3-2K: use /video-generation-v2-create [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-create), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create)
- H3-Context-IR：use /video-generation-v2-h3-context-ir [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-h3-context-ir)
- H3-Regenerate-2K：use /video-generation-v2-regeneration [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-regeneration), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-regeneration)


The examples below encode local H3\-Base output files as Base64 Data URLs\. For production use, uploading the video to a publicly accessible URL and passing that URL as `base_video` is recommended\.

For each case below, we provide reference outputs at both 2K and 768p generated directly through the Open Platform API, making it easier to validate the results\.

#### case\-T2VA

- Type: Text-to-video
- Duration: 10 seconds
- Aspect ratio: 16:9

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-t2va-h3-context-ir.sh">View script</a></td><td><pre><code class="language-json">{
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet&#39;s massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain&#39;s face and shakes strongly. The brilliant blue-white light from the fleet&#39;s gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.\noverall_soundscape: A low, resonant hum of the ship&#39;s ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge&#39;s bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.\nnon_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump.&quot;
    },
    &quot;duration&quot;: 10,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 8565,
      &quot;prompt_tokens&quot;: 5650,
      &quot;completion_tokens&quot;: 2915
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-t2va-h3-base.sh">View script</a></td><td><a href="assets/t2va.mp4">t2va.mp4</a></td></tr>
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-t2va-h3-regenerate-2k.sh">View script</a></td><td><a href="assets/t2va_2k.mp4">t2va_2k.mp4</a></td></tr>
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-t2va-reference-2k-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/h3_direct_2k.mp4">h3_direct_2k.mp4</a></td></tr>
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/h3_direct_768p.mp4">h3_direct_768p.mp4</a><br></td></tr>
  </tbody>
</table>

#### case\-I2VA

- Type: First-frame image-to-video
- Duration: 8 seconds
- Aspect ratio: adaptive

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-i2va-h3-context-ir.sh">View script</a></td><td><pre><code class="language-json">{
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;For the target video, at 0.00 seconds into the target video, &lt;Picture 1&gt; (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] This is a live-action, cinematic shot with a shallow depth of field. The camera holds a perfectly static shot throughout the entire eight-second duration, capturing a cozy family gathering in a traditional Japanese dining room. The scene opens with a large, intricately patterned blue and white ceramic bowl of ramen in the immediate foreground, rendered in crisp, sharp focus. The bowl sits on a smooth, polished long wooden table. Inside the bowl, a rich, oily golden-brown broth surrounds yellow wavy noodles, topped with two thick, round slices of chashu pork featuring visible fat marbling and a distinct spiral meat pattern. A generous mound of freshly chopped, bright green scallions rests in the center, and a crisp, dark green rectangular sheet of nori seaweed is tucked into the right edge. To the left of the bowl, a pair of light brown wooden chopsticks rests horizontally on a small, dark rectangular chopstick rest, near a small cylindrical ceramic teacup with blue painted patterns. On the right side of the table, a spherical paper lantern with a ribbed bamboo frame sits on a black wooden base. In the background, a large family of seven is gathered around the table, initially appearing as a soft, blurred presence. Behind them, traditional Japanese sliding shoji screens with wooden lattice frames are open, revealing a bright outdoor scene with lush green trees. Early in the clip, the thick, white steam rising from the hot ramen broth immediately intensifies, billowing upwards in thick, swirling clouds that dance continuously above the bowl. As the clip progresses into the middle seconds, the camera maintains its static position while the focus begins a deliberate, smooth shift deeper into the room. The foreground ramen bowl, its vibrant ingredients, and the rising steam gradually soften into a hazy, out-of-focus blur. Simultaneously, the family members in the background come into sharp, detailed clarity. The heavy steam continues to rise from the foreground, creating a dynamic, translucent veil between the camera and the family. With the focus now firmly locked on the background, the vibrant family dinner comes alive. The man in the dark navy blue long-sleeved shirt on the left leans forward, his mouth moving animatedly in a silent exchange. The young girl in the crisp white short-sleeved t-shirt beside him smiles brightly, looking toward the center of the table. The woman on the far left, wearing a soft light blue long-sleeved blouse, turns her head slightly, smiling gently. Across the table, the woman in the light grey button-down shirt smiles broadly, her eyes crinkling, as she rests her hands near her plate. The woman in the dark grey top further back uses her wooden chopsticks to pick up a small piece of food from a central ceramic dish filled with bright red pickled vegetables. The woman in the center back in the light grey sweater smiles gently, her hands clasped softly in front of her, observing the interaction. Throughout the remainder of the clip, the family continues their lively physical interaction, their mouths moving in continuous, silent cadences of conversation, while the thick, white steam from the blurred ramen bowl in the foreground never stops rising, adding a comforting atmosphere to the warm gathering.\n\noverall_soundscape: The soundscape begins with a quiet room tone mixed with the faint, airy rustle of the thick steam billowing from the hot ramen bowl in the foreground, accompanied by the subtle, continuous hissing and bubbling of the rich broth. As the visual focus shifts deeper into the room, the physical sounds of the bustling family dinner become dominant in the foreground. The clear, sharp clinking of ceramic bowls and wooden chopsticks touching plates is clearly heard as the family members reach for food. This is followed by the faint, muffled thud of a cup being set down on the smooth wooden table, and the subtle, rhythmic rustle of cotton and wool clothing as the family members lean forward and gesture, perfectly capturing the lively, physical atmosphere of the shared meal.\n\nnon_diegetic_music: A gentle, heartwarming acoustic guitar melody plays softly in the background, accompanied by the subtle, resonant notes of a traditional Japanese koto. The music maintains a slow, comforting tempo that enhances the cozy, nostalgic, and joyful atmosphere of the family gathering.&quot;
    },
    &quot;duration&quot;: 8,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 22822,
      &quot;prompt_tokens&quot;: 12800,
      &quot;completion_tokens&quot;: 10022
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-i2va-h3-base.sh">View script</a></td><td><a href="assets/i2va.mp4">i2va.mp4</a></td></tr>
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-i2va-h3-regenerate-2k.sh">View script</a></td><td><a href="assets/i2va_2k.mp4">i2va_2k.mp4</a><br></td></tr>
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-i2va-reference-2k-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/i2va_direct_2k.mp4">i2va_direct_2k.mp4</a></td></tr>
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-i2va-reference-768p-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/i2va_direct_768p.mp4">i2va_direct_768p.mp4</a></td></tr>
  </tbody>
</table>

#### case\-Ref2VA

- Type: Multimodal reference-to-video (video + audio)
- Duration: 5 seconds
- Aspect ratio: adaptive

<table>
  <thead>
    <tr><th>stage</th><th>request</th><th>result</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-ref2va-h3-context-ir.sh">View script</a></td><td><pre><code class="language-json">{
  &quot;task&quot;: {
    &quot;id&quot;: &quot;&lt;task_id&gt;&quot;,
    &quot;model&quot;: &quot;MiniMax-H3&quot;,
    &quot;status&quot;: &quot;succeeded&quot;,
    &quot;created_at&quot;: &quot;&lt;created_at&gt;&quot;,
    &quot;updated_at&quot;: &quot;&lt;updated_at&gt;&quot;,
    &quot;content&quot;: {
      &quot;prompt&quot;: &quot;subject_definitions:\n&lt;Subject 1&gt; is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in &lt;Video 1&gt;.\n&lt;Video 1&gt; is the source video for the editing task.\n&lt;Audio 1&gt; is the synchronized audio track of &lt;Video 1&gt;, providing the background music.\n&lt;Audio 2&gt; is the voice timbre reference for &lt;Subject 1&gt;&#39;s voice, containing a spoken male voiceover.\n\nsummary:\n[video editing + audio reference + audio reuse] The target video is an edited version of &lt;Video 1&gt;. &lt;Subject 1&gt;, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates &lt;Subject 1&gt;&#39;s face to speak the user-provided dialogue. &lt;Audio 1&gt; is partially reused as the continuous background music, while the target references the calm male voice timbre of &lt;Audio 2&gt; for &lt;Subject 1&gt;&#39;s spoken lines.\n\nretention_analysis:\n&lt;Subject 1&gt; (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.\n&lt;Video 1&gt; (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.\n&lt;Audio 1&gt;: partially_copy - the atmospheric background music from &lt;Audio 1&gt; is reused in the target video, mixed beneath the newly added spoken dialogue.\n&lt;Audio 2&gt;: reference - the target audio references the male voice timbre from &lt;Audio 2&gt; to generate &lt;Subject 1&gt;&#39;s spoken dialogue.\n\ndetailed_description:\nThe target video is in realistic photographic style.\n[Shot 1] The shot begins from the source &lt;Video 1&gt;, showing &lt;Subject 1&gt;, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from &lt;Audio 1&gt; plays continuously throughout the scene. &lt;Subject 1&gt; physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from &lt;Audio 2&gt;. Looking thoughtfully forward, &lt;Subject 1&gt; (S1) speaks softly, &lt;d&gt;[English] Follow the wind, live free.&lt;/d&gt; As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. &lt;Subject 1&gt; (S1) continues his thought, &lt;d&gt;[English] Leave worries behind, enjoy the moment.&lt;/d&gt; Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb&#39;s fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.\n\noverall_soundscape:\nThe soundscape consists of the continuous, atmospheric background music from &lt;Audio 1&gt;, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of &lt;Audio 2&gt;.\n\nnon_diegetic_music:\nThe atmospheric, sustained background music from &lt;Audio 1&gt; is reused as the continuous score, playing quietly beneath the spoken dialogue.&quot;
    },
    &quot;duration&quot;: 5,
    &quot;usage&quot;: {
      &quot;total_tokens&quot;: 39299,
      &quot;prompt_tokens&quot;: 33323,
      &quot;completion_tokens&quot;: 5976
    },
    &quot;ratio&quot;: &quot;16:9&quot;,
    &quot;task_type&quot;: &quot;h3_context_ir&quot;,
    &quot;modality&quot;: &quot;text&quot;
  }
}</code></pre></td></tr>
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-ref2va-h3-base.sh">View script</a></td><td><a href="assets/r2va.mp4">r2va.mp4</a><br></td></tr>
    <tr><td>Reference 2K result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-ref2va-reference-2k-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/r2va_2k.mp4">r2va_2k.mp4</a></td></tr>
    <tr><td>H3 API 2K in Open Platform for reference</td><td><a href="scripts/readme/full-2k-ref2va-h3-api-2k-in-open-platform-for-reference.sh">View script</a></td><td><a href="assets/r2va_direct_2k.mp4">r2va_direct_2k.mp4</a><br></td></tr>
    <tr><td>Reference 768P result by directly calling Open Platform API</td><td><a href="scripts/readme/full-2k-ref2va-reference-768p-result-by-directly-calling-open-platform-api.sh">View script</a></td><td><a href="assets/r2va_direct_768p.mp4">r2va_direct_768p.mp4</a><br></td></tr>
  </tbody>
</table>

### Prompting Guidance

Prompting guidance documents from the HuggingFace release are not copied into this repository to keep the markdown layout minimal.



## License

MiniMax H3 is released under the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).

## Contact Us

Contact us at [model@minimax.io](mailto:model@minimax.io).

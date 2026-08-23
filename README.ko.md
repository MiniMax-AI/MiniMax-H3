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

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ko.md"><strong>한국어</strong></a> |
  <a href="README.ja.md">日本語</a>
</p>

# MiniMax H3

## 에이전트 스킬

이 저장소에는 열 개의 스킬이 포함되어 있습니다. H3 프롬프트 작성 스킬은 다음 명령으로 설치합니다:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
```

이 스킬은 `skills/h3-prompt-writing/references/` 아래에 두 개의 프롬프트 가이드를 제공합니다. `base-en.txt`는 텍스트/키프레임 모드용이고, `ref-en.txt`는 전체 참조(Ref2VA) 모드용입니다.

**에이전트 호환성:** `h3-prompt-writing`은 외부 API 호출이 없는 순수 Markdown + 참조 파일 스킬이므로 Claude Code, Claude Agent SDK, Cursor, Windsurf, OpenAI 기반 에이전트/Codex, LangChain 등 `SKILL.md`와 로컬 파일을 읽을 수 있는 모든 환경에서 동작합니다. 함께 제공되는 `skills/h3-prompt-writing/agents/openai.yaml`은 [OpenAI의 스킬 사양](https://learn.chatgpt.com/docs/build-skills)에 따라 ChatGPT/Codex 스킬 UI를 위한 선택적 UI 메타데이터(표시 이름, 설명, 기본 프롬프트)만 추가할 뿐, 이 스킬을 OpenAI 에이전트로 제한하지 않습니다.

출처를 추적할 수 있는 공개 X(Twitter) 게시물로 비디오 브리프를 만들려면 `x-to-h3-video`를 설치합니다:

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill x-to-h3-video
```

이 스킬은 Xquik에서 공개 게시물을 최대 20개 가져오고, 출처 원장을 만들며, 사실 주장과 의견을 구분해 MiniMax H3 프롬프트를 준비합니다. 보조 스크립트는 환경에서 `XQUIK_API_KEY`를 읽고 X 쓰기 작업은 수행하지 않습니다.

나머지 여덟 개는 MiniMax Hub의 캔버스 워크플로(`hub_generate_video`, `hub_generate_image`, 캔버스 노드, 선택 카드 등)를 위해 만들어진 스타일별 비디오 생성 스킬로, 범용 에이전트 환경으로 이식할 수 없습니다:

<table align="center">
  <tr>
    <td align="center"><img src="assets/minimalist-product-ad-generator.gif" alt="minimalist-product-ad-generator" width="240"><br><a href="skills/minimalist-product-ad-generator/SKILL.md">minimalist-product-ad-generator</a></td>
    <td align="center"><img src="assets/3d-animation-short-generator.gif" alt="3d-animation-short-generator" width="240"><br><a href="skills/3d-animation-short-generator/SKILL.md">3d-animation-short-generator</a></td>
    <td align="center"><img src="assets/papercraft-stop-motion-explainer.gif" alt="papercraft-stop-motion-explainer" width="240"><br><a href="skills/papercraft-stop-motion-explainer/SKILL.md">papercraft-stop-motion-explainer</a></td>
    <td align="center"><img src="assets/brand-promo-video-generator.gif" alt="brand-promo-video-generator" width="240"><br><a href="skills/brand-promo-video-generator/SKILL.md">brand-promo-video-generator</a></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/music-video-subtitle-generator.gif" alt="music-video-subtitle-generator"          width="240"><br><a href="skills/music-video-subtitle-generator/SKILL.md">music-video-subtitle-generator</a></td>
    <td align="center"><img src="assets/co-op-game-intro-generator.gif" alt="co-op-game-intro-generator" width="240"><br><a href="skills/co-op-game-intro-generator/SKILL.md">co-op-game-intro-generator</a></td>
    <td align="center"><img src="assets/paper-collage-explainer-generator.gif" alt="paper-collage-explainer-generator" width="240"><br><a href="skills/paper-collage-explainer-generator/SKILL.md">paper-collage-explainer-generator</a></td>
    <td align="center"><img src="assets/handdrawn-live-video-generator.gif" alt="handdrawn-live-video-generator" width="240"><br><a href="skills/handdrawn-live-video-generator/SKILL.md">handdrawn-live-video-generator</a></td>
  </tr>
</table>

## 온라인 API
API를 통해 MiniMax\-H3를 직접 사용할 수 있습니다.
- Global: [platform\.minimax\.io](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) \| CN: [platform\.minimaxi\.com](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create)

## 온라인 앱
앱을 통해 MiniMax\-H3를 직접 사용할 수 있습니다.
- WebApp Global: [hailuoai\.video](https://hailuoai.video/tools/minimax-h3) \| CN: [hailuoai\.com](https://hailuoai.com/)
- Desktop Global: [hub\.minimax\.io](https://hub.minimax.io/) \| CN: [hub\.minimaxi\.com](https://hub.minimaxi.com/)


## 시스템 개요
MiniMax H3는 범용 옴니모달 생성 시스템입니다. 텍스트, 이미지, 비디오, 오디오로 구성된 멀티모달 컨텍스트를 통합적으로 이해하며, 최대 2K 해상도와 최대 15초 길이의 네이티브 스테레오 오디오 포함 비디오를 생성할 수 있습니다. 작업 일반화를 지향하는 시스템 설계 덕분에 H3는 사전 학습 단계에서 이미 폭넓은 멀티모달 컨텍스트 이해 및 생성 능력을 갖추고 있으며, 복잡한 멀티모달 지시를 따르는 데 뛰어난 성능을 보입니다.

H3는 다음 입력 및 출력 사양을 지원합니다:

| 범주 | 사양 |
|---|---|
| 출력 길이 | 4-15초 |
| 출력 화면비 | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 등을 포함한 다양한 화면비 지원 |
| 출력 해상도 | 다양한 해상도 지원. 기본적으로 짧은 변은 768픽셀로 설정됩니다. H3-Regenerate-2K를 통해 2K 생성을 수행할 수 있습니다 |
| 출력 프레임레이트 | 24 FPS |
| 출력 오디오 | 32 kHz 스테레오 |
| 지원 대화 언어 | 아랍어, 중국어, 영어, 프랑스어, 독일어, 이탈리아어, 일본어, 한국어, 포르투갈어, 러시아어, 스페인어 등 11개 언어를 안정적으로 지원합니다. 그 외 언어도 일정 수준 지원됩니다 |

### 모델 변형 및 입력 사양

| 모델 변형 | 입력 모드 | 사양 |
|---|---|---|
| H3-Base-FL2VA | First-and-last-frame mode | Supports zero, one, or two input images. <br><br>- No image input: Text-to-video mode <br>- One image input: First-frame-to-video or last-frame-to-video generation <br>- Two image inputs: First-and-last-frame-to-video generation |
| H3-Base-Ref2VA | Omni-reference mode | Supports multi-modal reference inputs: <br><br>- **Images:** ≤ 9 images <br>- **Videos:** ≤ 3 clips; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Audio:** ≤ 3 clips; audio must be accompanied by image or video input and cannot be used as the sole input; each clip must be 2–15 seconds long; total duration ≤ 15 seconds <br>- **Mixed inputs:** Maximum number of files across all input types is 12 |

![Image](assets/overview.png)

전체 H3 시스템은 다음 세 모듈로 구성됩니다:
- H3-Context-IR: As inputs become increasingly complex, we build a dedicated system to deeply understand and refine the input multimodal instructions, then convert them into a form that H3 can readily understand—the Context Intermediate Representation—for generation. **H3-Context-IR is critical to the quality of the final output, so we strongly recommend incorporating it into your generation pipeline or following the “Prompting Guidance” to build your own context-processing system.**
- H3-Base: Generates audio and video based on the H3-Context-IR output, producing results at 768p resolution.
- H3-Regenerate-2K: Feeds the 768p result together with the original context back into H3 to regenerate the output at 2K resolution. This process leverages both H3’s powerful generative capabilities and the rich information contained in the original context, enabling it to produce high-resolution outputs with more accurate details and greater visual fidelity.

## 모델 아키텍처

### H3\-Context\-IR

H3\-Context\-IR은 자유 형식의 멀티모달 입력을 위해 설계된 호스팅 기반 전처리 및 오케스트레이션 시스템입니다.

텍스트, 이미지, 오디오, 참조 비디오 사이의 관계와 이러한 자료가 목표 생성 결과와 어떻게 연결되는지를 해석합니다. 내부 워크플로에는 지시문 파싱, 크로스모달 연결, 시간적 이해, 복잡한 논리 추론이 포함됩니다.

H3\-Context\-IR은 컨텍스트에 대한 이해를 H3\-Base가 받아들일 수 있는 구조화된 표현으로 직렬화합니다. 사용자의 원래 의도에서 벗어나지 않는 범위에서 누락되었거나 충분히 지정되지 않은 의미 정보를 적절히 보완할 수도 있습니다.

H3\-Context\-IR은 다단계 워크플로와 여러 호스팅 모델 및 서비스에 의존하므로 이번 오픈소스 릴리스에는 포함되지 않습니다. 공식 워크플로의 동작을 재현할 수 있는 API를 제공하며, 개발자가 **프롬프트 가이드**를 따라 자체 전처리 시스템을 구축할 수 있도록 자세한 튜토리얼도 제공합니다.

자세한 사용 방법은 **권장 워크플로 - 전체 2K 워크플로**를 참고하세요.

**안전 가드레일**

사용자가 제출한 텍스트, 이미지, 비디오 및 향상된 프롬프트는 자동 검토 대상입니다. 불법, 음란물 또는 제3자 권리 침해가 의심되는 콘텐츠는 차단될 수 있습니다. 업계 표준 필터링 조치를 사용하지만 오탐과 미탐을 완전히 제거할 수는 없습니다. 이러한 가드레일은 MiniMax H3 Community License에 따른 라이선시의 의무, 특히 합법적 사용 및 사용 제한 관련 의무에 영향을 주지 않습니다.

### H3\-Base

![Image](assets/full-arch.png)

#### 아키텍처 개요

- H3\-Base는 각 모달리티를 해당 인코더 또는 VAE로 인코딩하고, 인코딩된 표현을 하나의 패킹된 멀티모달 시퀀스로 구성합니다. 전체 시퀀스가 H3\-Omni\-Transformer로 전달되기 전에 RoPE를 사용해 토큰 간의 필요한 공간 및 시간 관계를 포착합니다.

- 구체적으로 텍스트는 H3\-Encoder가 인코딩하고, 시각 입력은 H3\-Encoder와 H3\-VisualVAE가 함께 인코딩하며, 오디오는 H3\-AudioVAE만으로 인코딩합니다.

- H3\-Omni\-Transformer는 비디오와 오디오 latent를 공동으로 예측하며, 이후 각각 비디오와 스테레오 오디오로 디코딩됩니다.

- 긴 멀티모달 시퀀스의 계산 비용을 줄이기 위해 H3는 sparse-attention 학습과 추론을 네이티브로 지원합니다. 초기 오픈소스 릴리스는 full attention 추론만 제공합니다. sparse-attention 구현은 향후 업데이트에서 공개할 예정입니다.

#### H3\-Encoder

- H3\-Encoder는 Qwen3\-VL\-32B의 전체 사전 학습 가중치를 사용하며, 50번째 레이어의 hidden state를 H3\-Omni\-Transformer에 제공합니다.

- tokenizer 설정에는 `<d>` 같은 여러 특수 토큰을 추가했습니다. H3를 사용할 때는 H3 저장소에서 제공하는 tokenizer와 관련 설정 파일이 필요합니다.

#### H3\-VAE

H3는 시각 및 오디오 모달리티를 각각 별도의 latent로 표현합니다.

##### H3\-VisualVAE

- H3\-VisualVAE는 공간 압축 계수 16×, 시간 압축 계수 4×, 24개 latent 채널을 갖는 시간 인과적 비디오 오토인코더이며 f16t4d24로 표기합니다. 여러 latent 공간 최적화 기법을 적용해 재구성 품질과 latent 학습 용이성을 함께 개선합니다.

- H3\-Omni\-Transformer로 전달되기 전에 시각 latent는 `(time, height, width)` 차원에서 `1 × 2 × 2` 패치 크기로 추가 patchify됩니다. 따라서 Transformer에 입력되는 시각 토큰은 유효 공간 다운샘플링 계수 32×를 가지며, 시간 다운샘플링 계수는 4×로 유지됩니다.

- H3\-VisualVAE의 latent 공간은 재구성 품질과 생성 모델의 학습 용이성을 모두 고려해 최적화됩니다. 인코더를 학습한 뒤 디코딩 비용을 줄이고 재구성 품질을 더 높이기 위해 ViT 기반 디코더를 추가로 학습합니다.

##### H3\-AudioVAE

- H3-AudioVAE는 좌우 오디오 채널에 동일한 인코더와 디코더를 사용하되 각 채널을 독립적으로 처리합니다. 디코딩된 채널은 다시 결합되어 스테레오 오디오 입력과 출력을 가능하게 합니다.
- 각 채널에서 H3-AudioVAE는 32 kHz 오디오를 시간율 40 Hz의 latent token 시퀀스로 압축합니다.
- VA-VAE에서 영감을 받아, 오디오 재구성 품질을 유지하면서 생성 모델이 더 쉽게 학습할 수 있도록 latent 공간을 최적화합니다.

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



## 권장 워크플로

커뮤니티가 MiniMax H3를 올바르게 배포할 수 있도록 두 가지 검증 방법을 제공합니다.

전체 H3 시스템은 H3\-Context\-IR, H3\-Base, H3\-Regenerate\-2K 세 모듈로 구성되므로, “전체 2K 워크플로”는 Open Platform API와 로컬 배포 H3\-Base를 결합해 2K 출력을 검증하는 엔드투엔드 파이프라인을 제공합니다. “H3\-Base 로컬 배포” 섹션은 로컬 배포 H3\-Base만으로 768p 출력을 검증하는 방법을 제공합니다.

또한 “프롬프트 가이드” 섹션은 커뮤니티가 자체 프롬프트 시스템을 개발할 수 있도록 자세한 튜토리얼을 제공합니다.

### H3\-Base 로컬 배포

MiniMax H3는 두 개의 작업별 checkpoint로 공개됩니다. 각 checkpoint에는 전용 Omni Transformer Model과 필요한 processor, tokenizer, text encoder, Visual VAE, 독립 Audio VAE 구성 요소가 포함됩니다.

|Checkpoint|Supported Tasks|Input Conditions|Output|Precision|
|---|---|---|---|---|
|MiniMax\-H3 Base FL2VA|Text\-to\-Audio\-Video \(`t2va`\), First/Last\-Frame\-to\-Audio\-Video \(`fl2va`\)|Text; optional first frame, last frame, or both|Video and audio|BF16|
|MiniMax\-H3 Base Ref2VA|Reference\-to\-Audio\-Video \(`ref2va`\)|Text with reference images, videos, and/or audio|Video and audio|BF16|

공개된 checkpoint는 CFG 증류된 Omni Transformer 모델 가중치입니다. 

각 checkpoint는 다음 구성 요소를 포함하는 자체 완결형 Hugging Face 스타일 저장소로 배포됩니다:

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

모델을 다운로드합니다. 저장소에는 원본 checkpoint(`FL2VA/`, `Ref2VA/`)와 diffusers 형식이 함께 제공되므로 사용하는 프레임워크에 필요한 범위만 다운로드하세요:

`model_index.json`은 저장소 수준의 공개 진입점입니다. 작업군별 diffusers 인덱스는 `FL2VA/model_index.json` 및 `Ref2VA/model_index.json` 아래에 유지됩니다.

```bash
# Original checkpoint, both task families (SGLang, vLLM):
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" "Ref2VA/*" --local-dir MiniMax-H3

# Or a single task family:
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

diffusers 사용자는 수동 다운로드가 필요하지 않습니다. `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")`가 필요한 구성 요소만 가져옵니다. 로딩 방법은 [diffusers documentation](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md)을 참고하세요.

모델 서빙에는 다음 추론 프레임워크를 권장합니다:

- [SGLang](https://docs.sglang.io/) \- see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) 

- [vLLM](https://github.com/vllm-project/vllm) \- see [vllm recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)

- [diffusers](https://github.com/huggingface/diffusers) \- see [diffusers docs](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md)

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) \- see  [Comfy tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3); use [R2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) / [T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)

#### Sglang 배포

여기서는 sglang을 배포 예시로 사용합니다. 추가 배포 설정은 [MiniMax\-H3 deployment guide](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#3-serve-minimax-h3)를 참고하세요.

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

#### 재현 가능한 768p 사례

다음 세 가지 사용 사례 T2VA, FL2VA, Ref2VA는 MiniMax\-H3 비디오-오디오 생성을 재현하는 방법을 보여줍니다.

| 사용 사례 | 요청 | 결과 |
|---|---|---|
| T2VA | [스크립트 보기](scripts/readme/reproducible-768p-t2va-request.sh) | [t2va.mp4](assets/t2va.mp4) |
| FL2VA | [스크립트 보기](scripts/readme/reproducible-768p-fl2va-request.sh) | [fl2va.mp4](assets/fl2va.mp4) |
| Ref2VA | [스크립트 보기](scripts/readme/reproducible-768p-ref2va-request.sh) | [ref2va.mp4](assets/ref2va.mp4) |

### 전체 2K 워크플로

이 섹션에서는 로컬에 배포한 SGLang 서비스와 공식 **H3\-Context\-IR** 및 **H3\-Regenerate\-2K** API를 결합해 MiniMax API로 직접 생성한 2K 비디오 품질을 재현하는 방법을 설명합니다.
시작하기 전에 SGLang 엔드포인트와 MiniMax API 자격 증명을 설정합니다:

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

MiniMax 플랫폼:

API 문서:
- Create H3-2K: use /video-generation-v2-create [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-create), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create)
- H3-Context-IR：use /video-generation-v2-h3-context-ir [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-h3-context-ir)
- H3-Regenerate-2K：use /video-generation-v2-regeneration [EN-docs](https://platform.minimax.io/docs/api-reference/video-generation-v2-regeneration), [CN-docs](https://platform.minimaxi.com/docs/api-reference/video-generation-v2-regeneration)


아래 예시는 로컬 H3\-Base 출력 파일을 Base64 Data URL로 인코딩합니다. 프로덕션에서는 비디오를 공개 접근 가능한 URL에 업로드하고 해당 URL을 `base_video`로 전달하는 것을 권장합니다.

아래 각 사례에는 Open Platform API를 통해 직접 생성한 2K 및 768p 참조 출력을 함께 제공하여 결과 검증을 쉽게 합니다.

#### case\-T2VA

- 유형: 텍스트-비디오
- 길이: 10초
- 화면비: 16:9

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-t2va-h3-context-ir.sh">스크립트 보기</a></td><td><pre><code class="language-json">{
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-t2va-h3-base.sh">스크립트 보기</a></td><td><a href="assets/t2va.mp4">t2va.mp4</a></td></tr>
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-t2va-h3-regenerate-2k.sh">스크립트 보기</a></td><td><a href="assets/t2va_2k.mp4">t2va_2k.mp4</a></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-t2va-reference-2k-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/h3_direct_2k.mp4">h3_direct_2k.mp4</a></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-t2va-reference-768p-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/h3_direct_768p.mp4">h3_direct_768p.mp4</a><br></td></tr>
  </tbody>
</table>

#### case\-I2VA

- 유형: 첫 프레임 이미지-비디오
- 길이: 8초
- 화면비: 자동 조정

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-i2va-h3-context-ir.sh">스크립트 보기</a></td><td><pre><code class="language-json">{
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-i2va-h3-base.sh">스크립트 보기</a></td><td><a href="assets/i2va.mp4">i2va.mp4</a></td></tr>
    <tr><td>H3-Regenerate-2K</td><td><a href="scripts/readme/full-2k-i2va-h3-regenerate-2k.sh">스크립트 보기</a></td><td><a href="assets/i2va_2k.mp4">i2va_2k.mp4</a><br></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-i2va-reference-2k-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/i2va_direct_2k.mp4">i2va_direct_2k.mp4</a></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-i2va-reference-768p-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/i2va_direct_768p.mp4">i2va_direct_768p.mp4</a></td></tr>
  </tbody>
</table>

#### case\-Ref2VA

- 유형: 멀티모달 참조-비디오(비디오 + 오디오)
- 길이: 5초
- 화면비: 자동 조정

<table>
  <thead>
    <tr><th>단계</th><th>요청</th><th>결과</th></tr>
  </thead>
  <tbody>
    <tr><td>H3-Context-IR</td><td><a href="scripts/readme/full-2k-ref2va-h3-context-ir.sh">스크립트 보기</a></td><td><pre><code class="language-json">{
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
    <tr><td>H3-Base</td><td><a href="scripts/readme/full-2k-ref2va-h3-base.sh">스크립트 보기</a></td><td><a href="assets/r2va.mp4">r2va.mp4</a><br></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 2K 참조 결과</td><td><a href="scripts/readme/full-2k-ref2va-reference-2k-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/r2va_2k.mp4">r2va_2k.mp4</a></td></tr>
    <tr><td>참조용 Open Platform H3 API 2K 결과</td><td><a href="scripts/readme/full-2k-ref2va-h3-api-2k-in-open-platform-for-reference.sh">스크립트 보기</a></td><td><a href="assets/r2va_direct_2k.mp4">r2va_direct_2k.mp4</a><br></td></tr>
    <tr><td>Open Platform API 직접 호출로 생성한 768P 참조 결과</td><td><a href="scripts/readme/full-2k-ref2va-reference-768p-result-by-directly-calling-open-platform-api.sh">스크립트 보기</a></td><td><a href="assets/r2va_direct_768p.mp4">r2va_direct_768p.mp4</a><br></td></tr>
  </tbody>
</table>

### 프롬프트 가이드

Markdown 구성을 간결하게 유지하기 위해 Hugging Face 릴리스의 프롬프트 가이드 문서는 이 저장소에 복사하지 않았습니다.



## 라이선스

MiniMax H3는 [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)에 따라 배포됩니다.

## 문의

[model@minimax.io](mailto:model@minimax.io)로 문의해 주세요.

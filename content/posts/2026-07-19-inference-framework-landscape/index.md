---
title: "추론 최적화, 무엇을 선택할 것인가"
date: 2026-07-19
categories: [Engineering]
tags: [Inference, Optimization, PyTorch, LLM, Compiler]
draft: false
---

### 1. 추론 최적화 기술 개요

모델을 프로덕션 서비스로 운영하려면 추론 최적화 단계를 거치게 됩니다. 검증된 프레임워크나 라이브러리를 적용해보는 것이 자연스러운 접근법입니다. 바닥부터 커널을 새로 짜는 대신, 이미 여러 팀이 검증해 둔 컴파일러와 런타임을 활용하는 쪽이 안전하고 효율적입니다.

문제는 이 선택지의 목록이 빠르게 바뀐다는 점입니다. 2024년까지 PyTorch 진영에서 추론 최적화를 이야기할 때 TorchScript는 당연한 출발점이었습니다. 하지만 2026년 현재 PyTorch 진영의 지형은 크게 달라졌습니다(4절에서 다룹니다). 지금 시점에 어떤 선택지들이 있는지 한 번 정리해볼 필요가 있습니다.

이 글에서는 추론 최적화에 쓸 수 있는 기술들을 크게 세 갈래로 나누어 살펴봅니다. 범용 그래프 컴파일러·런타임, LLM 전용 서빙 엔진, 그리고 모바일·엣지에 특화된 런타임입니다. 이어서 PyTorch 자체의 배포 경로 변화와 벤더별 플랫폼 특화 옵션을 다룹니다. 마지막으로 NVIDIA·AMD·Apple Silicon·CPU·모바일 다섯 개 축으로 정리한 커버리지 매트릭스로 마무리합니다.

{{< svg name="diagram-1-stack.svg" caption="그림 1. 추론 최적화 도구를 세 갈래로 나누고, 그 아래 공통 하드웨어 기반을 두는 구도." >}}

### 2. 컴파일러 · 그래프 런타임

모델을 하나의 연산 그래프로 보고 하드웨어에 맞게 최적화된 형태로 컴파일하는 도구들이 있습니다. TensorRT [[4]](https://github.com/NVIDIA/TensorRT/releases) [[5]](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html), ONNX Runtime [[1]](https://github.com/microsoft/onnxruntime/releases), OpenVINO [[2]](https://github.com/openvinotoolkit/openvino/releases), Apache TVM [[6]](https://github.com/apache/tvm/releases), MLIR 기반의 IREE [[7]](https://github.com/iree-org/iree), 그리고 XLA와 IREE를 잇는 포터빌리티 레이어인 StableHLO [[8]](https://github.com/openxla/stablehlo)가 여기에 속합니다. (IREE와 StableHLO는 저 자신도 직접 다뤄본 적이 없어서 이번 정리는 공식 문서로 확인한 사실 위주로만 다룹니다.)

2026년 중반 기준으로 확인한 바로는 이 도구들 모두 유지보수가 활발합니다. ONNX Runtime은 거의 매달 릴리스되고 있습니다 [[1]](https://github.com/microsoft/onnxruntime/releases). OpenVINO는 6~8주 주기를 유지하면서 최근 llama.cpp용 백엔드를 프리뷰로 추가했는데, GGUF 모델을 Intel CPU·GPU·NPU에서 그대로 구동하는 경로입니다 [[2]](https://github.com/openvinotoolkit/openvino/releases) [[3]](https://www.phoronix.com/news/OpenVINO-2026.1-Released).

TensorRT는 별도의 릴리스 사이클로 계속 유지보수되고 있습니다 [[4]](https://github.com/NVIDIA/TensorRT/releases) [[5]](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html). Apache TVM도 릴리스마다 수십 건의 PR이 들어갈 만큼 개발이 이어지고 있는데 [[6]](https://github.com/apache/tvm/releases), 최근 체인지로그를 보면 CUDA 관련 항목이 대부분이고 ROCm·Apple 언급은 거의 없습니다. 엔지니어링 리소스가 어디에 쏠려 있는지 짐작할 수 있는 대목입니다. IREE는 문서 기준 CUDA·ROCm·Metal·Vulkan·CPU를 모두 지원합니다 [[7]](https://github.com/iree-org/iree). 아직 LF AI & Data Foundation의 sandbox 단계라는 점은 참고할 만합니다.

이 도구들의 공통점은 그래프 자체가 이전 출력에 따라 다음 계산의 형태를 스스로 바꾸지 않는다는 데 있습니다. 그래프가 정확히 한 번만 호출된다는 뜻은 아닙니다. 예를 들어 diffusion 모델은 같은 그래프를 여러 스텝에 걸쳐 반복 호출합니다. 입력 shape도 정적이거나 동적일 수 있습니다. 그 반복은 상위 오케스트레이션이 정해진 횟수만큼 호출하는 것일 뿐입니다. 모델 스스로 "다음에 얼마나 더 계산할지"를 실행 중에 결정하지는 않습니다. LLM의 decode 루프는 이 지점에서 다릅니다.

### 3. 범용 뉴럴 네트워크 vs LLM

2절에서 다룬 컴파일러들은 앞서 말한 전제, 즉 그래프 스스로 다음 계산을 동적으로 정하지 않는다는 전제 위에 있습니다. 이미지 분류든 음성 인식이든, 요청마다 계산량이 크게 다르지 않기 때문에 이 전제가 잘 맞습니다.

LLM 추론은 이 전제와 두 지점에서 어긋납니다.

첫째, LLM은 토큰을 하나씩 순차적으로 생성하는 autoregressive decoding 구조입니다. prompt를 한 번에 처리하는 prefill 단계는 compute-bound입니다. 반면 이후 토큰을 하나씩 뽑는 decode 단계는 연산량 자체는 작은데 메모리 접근이 지배적인 memory-bound 작업입니다. decode를 반복할 때마다 이전 토큰들의 attention 정보를 담은 KV cache가 시퀀스 길이에 비례해 계속 커집니다. 실행되는 동안 그래프의 상태(state) 자체가 계속 자라난다는 점에서 2절의 정적인 컴파일 그래프가 가정하는 것과는 다른 문제입니다.

둘째, 프로덕션 서빙에서는 요청이 동시에 도착하지도, 동시에 끝나지도 않습니다. 요청마다 도착 시점, 프롬프트 길이, 출력 길이가 다 다릅니다. 그래서 배치를 고정하지 않고 요청이 들고 나는 대로 GPU 위에서 계속 채워 넣는 continuous batching이 필요합니다.

{{< svg name="diagram-2-general-vs-llm.svg" caption="그림 2. 범용 뉴럴넷의 순전파 vs LLM의 autoregressive decoding · KV cache 증가 · continuous batching." >}}

이 두 가지(autoregressive decoding과 continuous batching)를 다루기 위해 만들어진 것이 vLLM, SGLang, TensorRT-LLM, llama.cpp 같은 LLM 전용 서빙 엔진입니다.

vLLM은 공식 문서 기준 NVIDIA CUDA뿐 아니라 AMD ROCm, Google TPU, Intel Gaudi·XPU까지 지원합니다 [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). Apple Silicon은 메인 설치 가이드에는 없고 vLLM-Metal이라는 서드파티 경로로만 닿을 수 있습니다 [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). 2026년 2분기 AMD 로드맵을 보면 신형 V1 엔진의 ROCm 이식도 아직 진행 중입니다 [[10]](https://github.com/vllm-project/vllm/issues/44092).

TensorRT-LLM은 컴파일 기반 엔진답게 트레이드오프가 뚜렷합니다. 한 벤치마크에 따르면 콜드스타트·빌드 시간은 vLLM·SGLang 대비 훨씬 깁니다(~28분 대 ~1분 안팎). 대신 H100 고동시성 시나리오에서는 오히려 TRT-LLM의 피크 처리량이 앞서기도 합니다 [[14]](https://leetllm.com/blog/llm-inference-engine-comparison-2026).

SGLang의 Apple Silicon 지원은 2026년 2월 로드맵 기준 전무한 상태에서 출발했습니다. PyTorch를 걷어내고 MLX로 바꾸는 실험이 진행 중이지만 같은 해 7월 기준으로도 핵심 서빙 커널은 미구현 상태입니다 [[13]](https://github.com/sgl-project/sglang/issues/19137). llama.cpp·GGML은 하루 단위로 릴리스될 만큼 개발 속도가 빠릅니다. CUDA·ROCm·Metal·Vulkan·SYCL까지 사실상 전 벤더 백엔드를 자체 구현하고 있습니다 [[15]](https://github.com/ggml-org/llama.cpp/releases). OpenVINO의 llama.cpp 백엔드나 Ollama의 llama.cpp 의존 등을 보면 [[3]](https://www.phoronix.com/news/OpenVINO-2026.1-Released) [[14]](https://leetllm.com/blog/llm-inference-engine-comparison-2026), GGUF가 크로스 엔진 교환 포맷처럼 자리잡는 흐름도 보입니다.

> **Disaggregated serving.** 서빙 엔진 위에 한 층 더 얹히는 오케스트레이션 흐름도 있습니다. NVIDIA Dynamo는 TensorRT-LLM의 후속이 아니라, PyTorch·SGLang·TensorRT-LLM·vLLM을 백엔드로 갈아끼울 수 있는 분산 서빙 레이어입니다 [[11]](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/). 핵심은 compute-bound인 prefill과 memory-bound인 decode를 물리적으로 다른 GPU·노드로 분리해서 각각 독립적으로 스케일하는 것입니다. 이것이 disaggregated serving입니다. NVIDIA Triton Inference Server는 사라진 게 아니라 "NVIDIA Dynamo-Triton"으로 이름을 바꿔 여전히 프로모션되고 있습니다 [[12]](https://developer.nvidia.com/triton-inference-server). 범용 추론은 Dynamo-Triton이, LLM 전용 최적화는 별도의 Dynamo가 맡는 구조입니다.

### 4. PyTorch 계열의 변화

PyTorch 진영에서는 그동안 추론 배포에 TorchScript를 쓰는 것이 사실상 기본값이었던 것 같습니다.

2024년 2월, PyTorch 코어 개발자는 dev-discuss 포럼에서 "TorchScript는 기술적으로 우월한 대체재가 없이는 deprecate하지 않는다"는 입장을 밝혔습니다 [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642). 이 시점 AOTInductor는 아직 프로토타입 단계였습니다 [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642).

2년 뒤, PyTorch 2.10에서 TorchScript는 공식적으로 deprecated 상태가 됐습니다. 2.11 릴리스 블로그는 "torch.export가 jit trace·script API를, ExecuTorch가 임베디드 런타임을 대체한다"고 명시하고 있습니다 [[17]](https://pytorch.org/blog/pytorch-2-11-release-blog/). GitHub 릴리스노트에는 torch.jit이 Python 3.14를 보장하지 않는다는 내용도 있습니다 [[19]](https://github.com/pytorch/pytorch/releases). PyTorch 공식 문서에서는 "TorchScript Unsupported PyTorch Constructs" 페이지가 삭제되고 torch.compiler API 문서로 리다이렉트되도록 바뀌었습니다 [[18]](https://docs.pytorch.org/docs/stable/jit_unsupported.html).

{{< svg name="diagram-3-pytorch-timeline.svg" caption="그림 3. TorchScript의 쇠퇴와 torch.export·AOTInductor·ExecuTorch·torch.compile의 부상." >}}

그 자리를 크게 세 갈래가 나눠 맡고 있습니다.

**torch.export**는 Python 런타임과 분리된 그래프를 만들어 다른 환경·언어에서도 로드·실행할 수 있게 합니다 [[20]](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html). `torch.jit.trace()`에는 없던 soundness(사이즈 관련 정수 연산까지 추적해 다른 입력에도 트레이스가 유효함을 보장하는 성질)을 명시적으로 내세웁니다.

**AOTInductor**는 이 torch.export 결과물만 입력으로 받는 TorchInductor의 특수판으로, 컴파일된 shared library(.so)를 만들어 C++ 같은 비-Python 프로덕션 환경에 배포합니다 [[21]](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). 다만 디바이스 간 이식성은 없습니다. CUDA용으로 컴파일한 `.so`는 CPU에서 실행할 수 없습니다 [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642).

**ExecuTorch**는 "임베디드 런타임"의 공식 후계자로 지정됐습니다. 하지만 범위가 그 이상으로 넓어지고 있습니다. 공식 문서 기준 XNNPACK(전 플랫폼 CPU), Metal·Core ML(Apple), Vulkan, Qualcomm·MediaTek·Arm 등 14개 백엔드를 지원합니다. 그중에는 Linux·Windows용 CUDA 백엔드도 포함됩니다 [[22]](https://docs.pytorch.org/executorch/stable/backends-overview.html). AMD ROCm 백엔드는 아직 없습니다.

**torch.compile·Inductor**는 이 변화들과 별개로 계속 확장하고 있습니다. Inductor는 기본적으로 OpenAI Triton을 코드 생성 백엔드로 사용합니다. 2.13.0에서는 여기에 CuTeDSL 기반의 두 번째 코드 생성 경로가 프로토타입으로 추가됐습니다. 같은 릴리스에서 Apple Silicon(MPS)에는 FlexAttention이 이식돼 sparse 패턴에서 SDPA 대비 최대 12배 스피드업을 보였습니다. ROCm은 AOTriton 0.12b로, Arm은 Armv9-A 타게팅으로 각각 확장됐습니다 [[19]](https://github.com/pytorch/pytorch/releases).

여기서 짚어둘 점은 torch.compile이 다루는 범위가 추론에 국한되지 않는다는 것입니다. torch.compile은 AOTAutograd를 통해 forward와 backward 그래프를 함께 캡처하기 때문에, 학습 단계의 backpropagation도 같은 컴파일 경로로 최적화됩니다. 이 글은 추론 관점에서 다루고 있습니다. 물론 PyTorch 2.x의 컴파일 스택 자체는 추론과 학습을 나누지 않는 하나의 아키텍처입니다.

TorchScript가 빠진 자리를 하나의 후계자가 아니라, 서버(torch.export + AOTInductor)·임베디드(ExecuTorch)·즉시 실행 최적화(torch.compile)라는 세 갈래가 나눠 맡은 셈입니다.

### 5. 벤더 · 플랫폼 특화 옵션

지금까지는 여러 벤더를 넘나드는 범용 도구를 다뤘습니다. 이번에는 특정 플랫폼 안에서 CPU·GPU·NPU 전 계층을 다루려는 벤더 특화 스택입니다.

Apple 진영에는 **Core ML**과 **MLX**가 상호보완적으로 존재합니다. Core ML은 iOS·macOS·watchOS·tvOS 전체에 배포할 수 있고 Neural Engine에 직접 접근하는 프로덕션 배포용입니다. MLX는 macOS Apple Silicon에서만 동작하고 모바일 배포는 되지 않는 대신 연구·파인튜닝에 맞춰져 있습니다 [[23]](https://cactuscompute.com/compare/coreml-vs-mlx). Apple이 공개한 M5 칩 실측 결과를 보면 이 구도가 좀 더 구체적으로 드러납니다. GPU Neural Accelerator는 MLX의 대형 행렬곱(연산 바운드, TTFT 구간)에서 최대 4배 빨라졌습니다. 반면 토큰 생성처럼 메모리 바운드인 구간에서는 1.2~1.3배 향상에 그쳤습니다 [[24]](https://machinelearning.apple.com/research/exploring-llms-mlx-m5). 같은 칩 세대라도 워크로드 성격에 따라 이득의 폭이 다르다는 걸 보여주는 수치입니다.

AMD 진영의 **MIGraphX**는 ONNX·TensorFlow 모델을 받아 AMD GPU(MIOpen·rocBLAS)와 CPU(DNNL·ZenDNN)용으로 최적화하는 그래프 컴파일러입니다 [[25]](https://rocm.docs.amd.com/en/docs-6.0.0/conceptual/ai-migraphx-optimization.html). ROCm 생태계의 최신 활력은 이 문서보다 3절에서 다룬 vLLM의 AMD 로드맵 쪽이 더 최근 신호입니다.

모바일·엣지 쪽은 **LiteRT**와 **MNN**이 대표적입니다. LiteRT는 2024년 TensorFlow Lite에서 이름을 바꾼 Google의 온디바이스 런타임입니다 [[26]](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/). TensorFlow 전용에서 벗어나 PyTorch·JAX 모델도 받아들이고 Google Tensor·Intel·MediaTek·Qualcomm의 NPU를 하나의 Compiled Model API로 추상화합니다 [[27]](https://github.com/google-ai-edge/litert). CUDA·ROCm 데스크톱 GPU 백엔드 없이 철저히 엣지에 집중합니다. Alibaba의 MNN은 CPU·GPU(Metal·Vulkan·CUDA)·NPU(CoreML·HIAI·NNAPI·QNN)까지 아우릅니다. MNN-LLM으로 온디바이스 LLM 서빙까지 영역을 넓히고 있습니다 [[28]](https://github.com/alibaba/MNN).

LiteRT·MNN·ExecuTorch가 공통으로 기대는 NPU 축 하나가 Qualcomm의 Hexagon입니다. **Qualcomm AI Engine Direct**(QNN)는 Kryo CPU·Adreno GPU·Hexagon NPU를 하나의 하드웨어 추상화 계층으로 묶습니다. TFLite·ONNX Runtime 같은 프레임워크는 이 계층에 위임(delegate)하는 방식으로 Hexagon NPU에 접근합니다 [[29]](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk). 6절 매트릭스의 "모바일·엣지" 열이 "확인됨"으로 표시된 항목 상당수는 실제로 이 QNN 계층을 거쳐 Snapdragon 위에서 돌아가는 셈입니다.

반대쪽 끝에는 AWS의 자체 실리콘인 **Trainium·Inferentia**가 있습니다. 둘 다 AWS Neuron SDK로 통합되어 있습니다. PyTorch·TensorFlow 코드를 거의 그대로 옮길 수 있다고 설명합니다 [[30]](https://aws.amazon.com/machine-learning/trainium/) [[31]](https://aws.amazon.com/machine-learning/inferentia/). AWS 측 자료는 vLLM·HuggingFace Transformers·TorchTitan을 네이티브로 지원한다고 밝힙니다 [[30]](https://aws.amazon.com/machine-learning/trainium/). 정작 vLLM 자체의 설치 문서에는 Neuron이 메인 하드웨어 목록이 아니라 서드파티 플러그인 쪽에 있습니다 [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). 이 글에서 다룬 프레임워크 대부분은 아직 이 실리콘까지는 닿지 않는다고 보는 편이 정확할 것 같습니다. vLLM 정도가 지금 확인되는 예외입니다.

{{< svg name="diagram-4-vendor.svg" caption="그림 4. Apple(Core ML·MLX), AMD(MIGraphX), 모바일·엣지(LiteRT·MNN) — 자기 플랫폼을 깊게 파는 세 축." >}}

### 6. 디바이스 커버리지 매트릭스

지금까지 살펴본 도구들을 NVIDIA GPU·AMD ROCm·Apple Silicon·범용 CPU·모바일·엣지, 다섯 개 축으로 정리하면 다음과 같습니다. "확인됨"은 공식 문서·릴리스노트로 근거를 확인한 경우입니다. "이번 조사 미확인"은 없다는 뜻이 아니라 이번 리서치에서 직접 확인하지 못했다는 뜻입니다.

{{< svg name="diagram-5-matrix.svg" caption="그림 5. 18개 프레임워크 × 5개 디바이스 축 커버리지 매트릭스 (2026-07 기준)." >}}

이 매트릭스는 스냅샷입니다. 반년만 지나도 몇몇 셀은 바뀌어 있을 겁니다. vLLM의 ROCm 지원이 V1 엔진까지 따라잡을 수도 있고, SGLang의 Apple 백엔드가 alpha를 벗어날 수도 있습니다. 중요한 건 이 매트릭스 자체를 외우는 것보다, 우리 팀의 워크로드가 걸쳐 있는 디바이스 조합 위에서 지금 무엇이 검증되고 있는지를 주기적으로 다시 확인하는 일입니다.

---

### References
*   [1] [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
*   [2] [OpenVINO Releases](https://github.com/openvinotoolkit/openvino/releases)
*   [3] [Phoronix — Intel Releases OpenVINO 2026.1 With Backend For Llama.cpp](https://www.phoronix.com/news/OpenVINO-2026.1-Released)
*   [4] [NVIDIA/TensorRT Releases (OSS plugins/parsers)](https://github.com/NVIDIA/TensorRT/releases)
*   [5] [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
*   [6] [Apache TVM Releases](https://github.com/apache/tvm/releases)
*   [7] [IREE: An MLIR-based end-to-end compiler and runtime](https://github.com/iree-org/iree)
*   [8] [OpenXLA StableHLO](https://github.com/openxla/stablehlo)
*   [9] [vLLM — GPU Installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
*   [10] [vLLM — AMD Development Roadmap (2026 Q2), Issue #44092](https://github.com/vllm-project/vllm/issues/44092)
*   [11] [NVIDIA Technical Blog — Introducing NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
*   [12] [NVIDIA Dynamo-Triton (formerly Triton Inference Server)](https://developer.nvidia.com/triton-inference-server)
*   [13] [SGLang — Apple Device Support (2026 Q2), Issue #19137](https://github.com/sgl-project/sglang/issues/19137)
*   [14] [vLLM vs SGLang vs TensorRT-LLM vs Ollama: Choosing an Inference Engine in 2026](https://leetllm.com/blog/llm-inference-engine-comparison-2026)
*   [15] [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
*   [16] [dev-discuss.pytorch.org — torch.export / torchserve / executorch / aotinductor](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642)
*   [17] [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
*   [18] [PyTorch Docs — TorchScript Unsupported PyTorch Constructs](https://docs.pytorch.org/docs/stable/jit_unsupported.html)
*   [19] [pytorch/pytorch Releases](https://github.com/pytorch/pytorch/releases)
*   [20] [PyTorch Docs — torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)
*   [21] [PyTorch Docs — AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
*   [22] [ExecuTorch Docs — Backends Overview](https://docs.pytorch.org/executorch/stable/backends-overview.html)
*   [23] [Core ML vs MLX: Apple's Two ML Frameworks Compared](https://cactuscompute.com/compare/coreml-vs-mlx)
*   [24] [Apple Machine Learning Research — Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
*   [25] [AMD ROCm Documentation — Inference Optimization with MIGraphX](https://rocm.docs.amd.com/en/docs-6.0.0/conceptual/ai-migraphx-optimization.html)
*   [26] [Google Developers Blog — TensorFlow Lite is now LiteRT](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)
*   [27] [google-ai-edge/litert](https://github.com/google-ai-edge/litert)
*   [28] [alibaba/MNN](https://github.com/alibaba/MNN)
*   [29] [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
*   [30] [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
*   [31] [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)

---
title: "GPU 최적화의 단위 변화: Thread에서 Tile로"
date: 2026-03-30
categories: [Engineering]
tags: [GPU, Optimization, Compiler, MLIR]
draft: false
---

### 1. 서론: 모델 수술(Model Surgery)의 피로도
현재 실무에서 ONNX Runtime이나 TensorRT [[1]](https://developer.nvidia.com/tensorrt)를 활용하여 추론 성능을 확보하기 위해 수행하는 가장 흔한 작업은 '모델 리라이팅(Model Rewriting)'입니다. TensorRT와 같은 추론 최적화 엔진이 지원하지 않는 연산을 대체하거나, TorchDynamo [[2]](https://pytorch.org/get-started/pytorch-2.0/)에서 Graph Break를 유발하는 `if` 분기문을 제거하기 위해 수학적 마스킹(Masking)을 적용하는 것이 대표적입니다. 이러한 작업은 정적인 연산 그래프를 강제하여 컴파일러의 최적화 효율을 높려는 엔지니어의 수동 개입이라 할 수 있습니다.

### 2. 제어 흐름(Control Flow)과 모델 수술의 미래
이러한 '수술'은 대개 리서치 단계의 코드가 프로덕션 환경의 제약 사항과 괴리가 있을 때 발생합니다. 다행히 최근 LLM이나 Diffusion 모델처럼 백본(Backbone) 아키텍처가 표준화되면서 과거보다 모델 수술의 빈도는 줄어들고 있으나, 여전히 최첨단 리서치 모델을 이식하는 과정에서 엔지니어는 컴파일러의 비위를 맞추는 작업을 반복하곤 합니다. 차세대 추상화 기술들이 안착한다면, 엔지니어가 직접 코드를 리라이팅하는 대신 컴파일러의 미들엔드(Middle-end)가 이를 더 영리하게 처리하게 될 것으로 기대합니다.

### 3. 최적화 단위의 전이: Triton과 CUTLASS CuTe
최적화의 패러다임은 이미 개별 스레드(Thread) 제어에서 블록(Block) 및 타일(Tile) 단위의 추상화로 이동하고 있습니다.

*   **Triton** [[3]](https://github.com/triton-lang/triton): 루프 분석의 복잡성을 피하기 위해 사용자에게 처음부터 블록 단위 코딩을 제안합니다. 내부적으로는 정수 분석(Integer Analysis) 및 기호적 최적화 패스를 통해 GPU 하드웨어 제약(뱅크 컨플릭트 등)에 맞는 인덱싱을 자동 생성합니다. 이는 과거 폴리헤드럴(Polyhedral) 컴파일러가 지향했던 루프 최적화의 이점을 현대적인 타일 기반 아키텍처로 가져온 결과입니다.
*   **CUTLASS CuTe** [[4]](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quick_start.md): CUTLASS 3.0부터 도입된 **CuTe**는 데이터의 '레이아웃(Layout)'과 '타일(Tile)'을 수학적으로 추상화합니다. 스레드 계층 구조(Hierarchy)에 종속된 수동 인덱싱 대신, 데이터의 형태(Shape)와 스트라이드(Stride)를 하드웨어의 계층적 자원에 매핑하는 **Composability**에 집중합니다. 이를 통해 엔지니어는 복잡한 하드웨어 명령어(TMA, WGMMA 등)를 직접 다루지 않고도 고성능 커널을 설계할 수 있는 기반을 얻습니다.

### 4. 차세대 추상화: JAX Pallas, Tunix, 그리고 CUDA Tile
최근 논의되는 기술들은 이러한 타일 단위 최적화를 더 높은 수준의 컴파일러 스택으로 끌어올리려는 움직임을 보여줍니다.

*   **JAX Pallas** [[5]](https://jax.readthedocs.io/en/latest/pallas/index.html) & **Tunix** [[6]](https://github.com/google/tunix): 구글은 Pallas를 통해 '타일' 중심의 통합 커널 언어를 제공함과 동시에, 그 배후에서 **Tunix**와 같은 MLIR [[7]](https://mlir.llvm.org/) 기반의 새로운 컴파일러 인프라를 구축하고 있습니다. Tunix는 TPU와 GPU 모두를 타겟으로 하며, MLIR 다이얼렉트(Dialect)를 활용해 하드웨어 특성에 맞는 타일링과 마이크로 커널 생성을 자동화하려 시도합니다. 
*   **CUDA Tile (Tile IR)** [[8]](https://github.com/NVIDIA/cuda-tile): 엔비디아가 공개한 새로운 프로그래밍 모델로, 기존의 스레드 중심(SIMT) 모델에서 벗어나 **'타일'을 연산의 기본 단위(Primitive)**로 격상시켰습니다. **Tile IR**이라 불리는 MLIR 다이얼렉트를 통해 하드웨어의 메모리 계층과 텐서 코어 사양을 추상화하며, 엔지니어가 직접 모델을 리라이팅하는 대신 컴파일러가 Blackwell [[9]](https://www.nvidia.com/en-us/data-center/blackwell/) 등의 차세대 하드웨어 스펙에 맞춰 최적의 타일 레이아웃과 데이터 흐름을 생성하는 구조를 지향하고 있습니다.

### 5. 결론: '오토-최적화' 시대, 엔지니어의 스탠스에 대한 제언
`torch.compile`과 같은 기술이 안착하고 코딩 에이전트(Claude Code, Gemini 등)가 최적화 패턴을 학습하여 커널을 생성하는 시대가 오면, 엔지니어링의 역할은 다음과 같이 변모할 것으로 조심스럽게 예측해 본다.

*   **리서치 엔지니어**: 모델 설계 단계에서 하드웨어 친화적(Hardware-aware)인 아키텍처를 고민해야 할 필요성이 커질 것입니다. 다만 이는 개인의 완벽한 숙련보다는 **조직 내 최적화 전문가와의 협업, 사내 지식 베이스, 혹은 AI 에이전트의 가이드**를 적시에 활용하여 모델의 수학적 구조를 최적화 도구들이 이해하기 쉽게 설계하는 방향이 될 것입니다.
*   **인프라/엔터프라이즈 엔지니어**: 개별 모델을 수동으로 고치기보다 거대한 인프라 전체의 효율을 관리하는 데 집중하게 될 것입니다. 수천 개의 모델에 공통으로 적용될 수 있는 **컴파일러 정책(Policy)과 표준화된 최적화 패스**를 설계하여, AI 에이전트와 컴파일러가 최상의 성능을 낼 수 있는 '운동장'을 만드는 역할입니다.
*   **최적화 전문가**: 단순한 API 호환성 확보를 위한 리라이팅은 줄어들겠지만, 두 가지 핵심 영역에서의 역할은 오히려 심화될 것으로 보입니다. 
    1.  **심층 수술(Deep Surgery)**: 하드웨어의 한계 성능을 끌어내기 위해 컴파일러가 자동화하지 못하는 복잡한 데이터 의존성을 해결하고, 모델의 레이아웃 자체를 최신 하드웨어 명령어(Intrinsics)에 맞게 재설계하는 고도의 엔지니어링 작업입니다. 
    2.  **프리미티브 정의(Defining Primitives)**: 기존 AI 워크로드와는 전혀 다른 데이터 특성을 가진 **미개척 도메인의 새로운 연산 프리미티브**를 최초로 정의하는 것입니다. 코딩 에이전트조차 참고할 데이터가 없는 영역에서 최적화 경로(Path)를 개척하고 이를 컴파일러 다이얼렉트로 설계하는 역량이 핵심이 될 것입니다.

결국 모델을 수동으로 고치는 단순 '수술'의 비중은 줄어들겠지만, 컴파일러 스택과 AI 에이전트를 도구로 활용해 **비즈니스에 최적화된 시스템을 오케스트레이션**하는 역량은 더욱 가치 있어질 것입니다. 로우레벨의 스레드 제어를 넘어, 하드웨어 자원의 물리적 특성을 거시적인 시스템 최적화 전략으로 연결하는 아키텍트로서의 관점 전환이 더욱 중요해질 것으로 보입니다.

---

### References
*   [1] [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
*   [2] [PyTorch 2.0: TorchDynamo and torch.compile](https://pytorch.org/get-started/pytorch-2.0/)
*   [3] [Triton: An Intermediate Language and Compiler for GPU Programming](https://github.com/triton-lang/triton)
*   [4] [NVIDIA CuTe: Layout-centric programming for CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quick_start.md)
*   [5] [JAX Pallas: A Kernel Language for JAX](https://jax.readthedocs.io/en/latest/pallas/index.html)
*   [6] [Google Tunix: MLIR-based Compiler Infrastructure](https://github.com/google/tunix)
*   [7] [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
*   [8] [NVIDIA CUDA Tile (Tile IR) and MLIR Dialects](https://github.com/NVIDIA/cuda-tile)
*   [9] [NVIDIA Blackwell Architecture Technical Overview](https://www.nvidia.com/en-us/data-center/blackwell/)

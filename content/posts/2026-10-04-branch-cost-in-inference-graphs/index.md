---
title: "컴파일드 추론 그래프에서 분기문 하나의 비용"
date: 2026-10-04T00:00:00+09:00
categories: [Engineering]
tags: [Inference, Compiler, CUDA, TensorRT, PyTorch]
description: "분기문을 정적 추론 그래프에 넣는 세 가지 방법을 CPU와 A10G GPU에서 측정하고 계산량, CUDA Graph 캡처, host 동기화 비용을 비교합니다."
summary: "분기문을 정적 추론 그래프에 넣는 세 가지 방법을 CPU와 A10G GPU에서 측정하고 계산량, CUDA Graph 캡처, host 동기화 비용을 비교합니다."
images: ["og/2026-10-04-branch-cost-in-inference-graphs-ko.png"]
draft: false
---

### 1. 분기문을 그래프 안에 넣어야 할 때

리서치 단계에서 PyTorch로 만든 모델을 서비스용 추론 엔진으로 옮기다 보면 자주 만나는 문제가 두 가지 있습니다. 하나는 입력 길이나 텐서 shape이 요청마다 달라지는 dynamic shape이고 다른 하나는 모델 안의 분기문입니다. 연구 코드에서는 파이썬 `if` 한 줄로 끝나던 일을 TensorRT나 ONNX처럼 그래프를 미리 고정하는 런타임으로는 그대로 옮길 수 없습니다. 엔진을 빌드하거나 export하는 순간 코드는 한 번 트레이싱되어 고정된 연산 그래프가 됩니다. 이때 `if`는 트레이싱할 때의 조건값 하나로 고정됩니다.

분기문은 그래프를 조건별로 나눠서 처리하는 편이 가장 깔끔합니다. 경로마다 그래프를 따로 떼어 엔진을 두 벌 만들고 실행할 때 어느 엔진을 부를지 host에서 고르면 됩니다. 각 엔진은 분기가 없는 정적 그래프라서 최적화하기도 쉽습니다.

문제는 분기가 루프 안에 있거나 개수가 많을 때입니다. 루프 안의 분기는 반복마다 다른 길을 탈 수 있어서 실행 전에 엔진 하나를 고를 수 없습니다. 분기가 여러 개면 그 조합마다 그래프가 필요해 엔진 수가 빠르게 늘어납니다. 이럴 때는 부득이 분기 자체를 그래프 안에 넣어 컴파일해야 합니다.

{{< svg name="fig1-two-ways.svg" caption="그림 1. 조건을 실행 전에 알면 엔진을 두 벌로 나눕니다. 분기가 루프 안에 있으면 분기 자체를 그래프에 넣어야 합니다." >}}

그래프 안의 분기는 조건의 단위에 따라 다시 둘로 나뉩니다. 배치 전체가 같은 길로 가는 스칼라 조건과, 배치 안의 행마다 길이 다른 행 단위 조건입니다. `route_id`가 0인 행은 `branch_fn_0`을, 1인 행은 `branch_fn_1`을 거쳐야 하는 레이어가 후자의 예입니다. 그런데 각 런타임의 조건부 연산자(TensorRT `IIfConditionalLayer`, ONNX `If`, XLA `Conditional`, `jax.lax.cond`, `torch.cond`)는 공식 문서상 모두 스칼라 조건만 받습니다. 실제로 행 단위 조건을 넣어 보면 이렇게 거부됩니다.

```
ONNX Runtime   Compute If nodes condition input must have exactly one element
JAX            TypeError: Pred must be a scalar, got ... shape (512,)
torch.compile  Detected data-dependent branching (graph_count=2, graph break 1회)
```

`torch.compile`은 그래프를 쪼갠 뒤 eager로 이어 붙이기 때문에 실행 자체는 됩니다. 하지만 그래프 하나를 통째로 내보내야 하는 ONNX나 TensorRT 배포에는 이런 우회로가 없습니다.

이 글에서는 분기를 그래프 안에 넣는 세 가지 방법을 맥북 CPU와 A10G GPU에서 비교한 결과를 정리합니다. 두 번 계산하는 손해가 얼마나 큰지, 조건이 스칼라일 때 진짜 분기가 더 빠른지, 깊은 파이프라인 한가운데에서는 무슨 일이 생기는지, host를 거치지 않고 분기할 수 있는지를 차례로 살펴봅니다. 반복 파이프라인에서는 계산량보다 host 동기화 여부가 더 중요했습니다.

### 2. 세 가지 방법

```python
def masked_dense(x, route_id, W0, b0, W1, b1):
    y0 = branch_fn_0(x, W0, b0)
    y1 = branch_fn_1(x, W1, b1)
    mask0 = (route_id == 0).to(x.dtype)[:, None]
    mask1 = (route_id == 1).to(x.dtype)[:, None]
    return y0 * mask0 + y1 * mask1

def gather_scatter(x, route_id, W0, b0, W1, b1):
    out = torch.empty_like(x)
    idx0 = (route_id == 0).nonzero(as_tuple=True)[0]
    idx1 = (route_id == 1).nonzero(as_tuple=True)[0]
    if idx0.numel() > 0:
        out.index_copy_(0, idx0, branch_fn_0(x.index_select(0, idx0), W0, b0))
    if idx1.numel() > 0:
        out.index_copy_(0, idx1, branch_fn_1(x.index_select(0, idx1), W1, b1))
    return out
```

{{< svg name="fig2-three-methods.svg" caption="그림 2. 세 가지 방법의 데이터 흐름. 아래 표시는 각 방법의 장점(파랑)과 비용(주황)입니다." >}}

**masked**는 모든 행을 두 경로로 다 계산한 뒤 마스크를 곱해 필요한 결과만 남깁니다. 조건값은 마스크 텐서의 숫자로만 흘러가므로 그래프 모양은 조건과 상관없이 늘 같습니다. 대신 GEMM을 두 번 해야 합니다. 두 경로를 다 실행하고 결과만 조건으로 고르는 방식을 predication이라고 부르기 때문에, 이 방식은 행 단위 predication(row-wise predication)이라고도 부릅니다.

**gather_scatter**는 `nonzero()`로 경로별 행 번호를 뽑고 그 행만 모아(`index_select`) 계산한 뒤 제자리에 돌려놓습니다(`index_copy_`). GEMM은 한 번이면 됩니다. 다만 `nonzero()`가 내놓는 텐서 길이는 `route_id` 값을 봐야 정해집니다.

**진짜 분기**(`If`, `torch.cond`)는 조건이 배치 전체에 하나일 때만 쓸 수 있습니다. 한쪽 경로만 계산하는 대신 어느 쪽으로 갈지 정하려면 누군가 조건값을 읽어야 합니다.

### 3. 두 번 계산하는 손해는 얼마나 큰가

먼저 맥북(M2 Max) CPU에서 PyTorch 2.13으로 행 단위 조건의 두 방법을 비교했습니다. N=4096, hidden_dim=1024에서 30회 평균을 냈고 세 구현의 결과가 일치하는지(rtol=1e-5)부터 확인했습니다. split은 `route_id`가 1인 행의 비율입니다.

| 구현 | split 0.5 | split 0.01 | gather_scatter 대비 |
|---|---|---|---|
| **gather_scatter** | **4.62 ms** | **5.05 ms** | 가장 빠름 |
| masked (eager) | 8.75 ms | 8.81 ms | 1.74~1.89배 느림 |
| masked (`torch.compile`) | 9.02 ms | 9.50 ms | 1.88~1.95배 느림 |

N=4096에서는 gather_scatter가 masked보다 1.74~1.95배 빨랐고 N=512에서도 약 1.4배 빨랐습니다. split 비율에 따른 차이는 거의 없었습니다. masked는 GEMM을 두 번 하는 만큼 시간이 더 걸렸습니다. 참고로 행마다 파이썬 `if`를 도는 loop_if는 계산량이 가장 적은데도 118.91 ms가 걸려 masked보다 약 14배 느렸습니다. 행마다 파이썬에서 연산을 호출하는 비용이 계산 절감분보다 컸습니다.

branch 하나의 계산량이 작으면 순서가 바뀝니다. N=4096, split 0.5에서 hidden_dim을 줄여 보니 32에서는 masked가 빨랐고(0.256 ms 대 0.353 ms), 128에서는 gather_scatter가 빨랐습니다(0.478 ms 대 0.563 ms). gather_scatter가 매번 추가로 내는 연산 6개(`nonzero`, `index_select`, `index_copy_` 각 2회)가 고정비로 붙기 때문입니다. 반대로 branch를 4개, 8개로 늘리면 masked는 18.78 ms, 37.92 ms로 branch 수에 비례해 늘었고 gather_scatter는 5.39 ms, 6.90 ms로 거의 그대로였습니다.

{{< svg name="fig3-cpu-cost.svg" caption="그림 3. CPU 실측. (a) hidden_dim 8과 32에서는 masked가, 128과 1024에서는 gather_scatter가 빨랐습니다. (a)는 hidden_dim을 바꿔 가며 측정한 별도 회차의 값이라 1024의 배율이 위 표와 조금 다릅니다. (b) branch 수가 늘면 masked의 시간만 비례해 늘어납니다." >}}

속도만 보면 gather_scatter가 유리합니다. 하지만 그래프로 옮기면 문제가 생깁니다. `torch._dynamo.explain`으로 보면 `nonzero()` 두 번에서 그래프가 끊겨 graph_count=3이 됩니다. ONNX export는 성공하지만 출력의 배치 축이 동적 차원 `n`으로 선언됩니다.

이 `n`은 1절에서 말한 dynamic shape과도 성격이 다릅니다. 입력 길이처럼 범위를 미리 최적화 프로파일에 등록해 두는 차원과 달리, 텐서 값을 봐야 크기가 정해집니다. TensorRT는 이런 연산을 Data-Dependent Shape로 따로 분류하고 직접 구현하려면 훨씬 까다로운 플러그인 경로를 타야 합니다. 그래서 정적 그래프로 배포할 때는 여전히 masked가 안전한 기본값이었습니다.

### 4. 조건이 배치 전체에 하나라면

조건이 스칼라라면 네이티브 조건부 연산자를 쓸 수 있습니다. 한 경로만 계산하므로 masked보다 빠를 것으로 예상했습니다. 측정 설정은 N=4096, hidden_dim=1024, cond=True입니다. CPU는 50회, A10G는 200회 평균입니다.

| 환경 | masked (둘 다 계산) | 진짜 분기 (한쪽만 계산) | 빠른 쪽 |
|---|---|---|---|
| CPU, 파이썬 `if` (eager) | 9.22 ms | **3.95 ms** | 진짜 분기 2.33배 |
| CPU, `torch.cond` (compile) | 9.22 ms | **4.23 ms** | 진짜 분기 2.18배 |
| CPU, ONNX Runtime `If` | **9.22 ms** | 15.16 ms | masked 1.64배 |
| A10G, TensorRT `If` | 0.693 ms | **0.425 ms** | 진짜 분기 1.63배 |

{{< svg name="fig4-scalar-ratio.svg" caption="그림 4. 스칼라 조건에서 진짜 분기와 masked의 속도 비교. 네 환경 중 ONNX Runtime에서만 masked가 빨랐습니다." >}}

PyTorch에서는 예상대로 진짜 분기가 2.18~2.33배 빨랐습니다. 그런데 ONNX Runtime의 `If`는 절반만 계산하는데도 가장 느렸습니다. 서브그래프를 실행하는 오버헤드가 절감분보다 큰 것으로 보입니다. 같은 패턴을 A10G의 TensorRT로 옮기자 이번에는 `If`가 1.63배 빨랐습니다. 네이티브 `If`가 항상 빠르지는 않았고 결과는 런타임 구현에 따라 달랐습니다.

여기서 궁금한 점이 하나 생겼습니다. GPU에서 계산된 조건값을 보고 다음 커널을 고르려면 누군가 그 값을 host로 읽어 와야 하지 않을까요? nsys로 20회 호출을 프로파일링해 보니 실제로 그랬습니다. `If` 엔진은 호출마다 정확히 한 번 1바이트를 D2H로 복사했고 masked 엔진은 한 번도 복사하지 않았습니다. masked 엔진의 GEMM 커널은 ncu로도 확인했습니다. 워프 안 스레드가 늘 같은 분기 대상으로 모였고(`branch_targets_threads_uniform` 100%) 커널 안에서 실행 경로가 갈라지지 않았습니다. 조건이 마스크 텐서의 값으로만 전달되기 때문입니다.

### 5. 파이프라인 한가운데에 있으면

지금까지는 매 호출 뒤 GPU를 완전히 비우는 고립된 측정이었습니다. 실제 모델에서는 같은 레이어가 수십 번 반복되고 CPU는 커널을 동기화 없이 미리 쌓아 두면서 launch 비용을 숨깁니다. 그 흐름 중간에서 host가 조건값을 기다려야 한다면 큐가 거기서 끊기지 않을까요? 1절에서 말한 루프 안의 분기가 바로 이 상황입니다.

{{< svg name="fig5-host-sync.svg" caption="그림 5. host 동기화가 비동기 실행을 끊는 과정. (a) 조건이 데이터로 흐르면 CPU는 커널을 미리 쌓아 두고 GPU는 쉬지 않고 실행합니다. (b) host가 조건값을 읽으려면 CPU는 앞선 커널과 D2H 복사가 끝날 때까지 기다립니다. 판단을 마친 뒤에야 다음 커널을 큐에 넣으므로 그사이 GPU에 유휴 구간이 생깁니다. 46.7~47.1 μs는 아래 표의 실측값입니다." >}}

CPU와 GPU가 동시에 일하던 흐름이 조건값 하나 때문에 서로를 기다리는 흐름으로 바뀝니다. 그동안 GPU는 일을 받지 못합니다.

이를 확인하려고 32개 레이어 중 16번만 조건부인 파이프라인(hidden_dim=512, N=2048)을 네 가지 방식으로 만들었습니다.

| 구현 | 컴파일 그래프 수 | CUDA Graph 캡처 | 16번 레이어 유휴 구간 | D2H 복사 | eager 실행 1회 |
|---|---|---|---|---|---|
| masked | 1 | 성공 | 없음 | 0 | 3.48 ms |
| gather_scatter | 3 | 실패 | 47.1 μs (중앙값의 61배) | 2 | 3.47 ms |
| scalar `if` | 1* | 실패 | 46.7 μs (중앙값의 61배) | 1 | 3.30 ms |
| `torch.cond` | 1 | 실패 | 측정 안 함 | 측정 안 함 | 3.31 ms |

컴파일 그래프 수는 맥북 CPU에서 `torch._dynamo.explain`으로, 나머지는 A10G에서 측정했습니다. *scalar `if`는 explain에서 1로 나오지만 실제로 컴파일하면 `Tensor.item()`에서 graph break 경고가 납니다.

{{< figure src="pipeline-timeline-3way.png" caption="그림 6. A10G에서 nsys로 기록한 커널 실행 구간. masked는 끊김 없이 이어지고 gather_scatter와 scalar if는 16번 레이어 위치에 GPU 유휴 구간이 생깁니다." >}}

먼저 CUDA Graph로 캡처된 구현은 masked뿐이었습니다. `torch.cond`는 `torch.compile`에서 그래프 하나로 잡혔는데도 캡처에 실패했습니다. eager 모드의 `torch.cond`도 결국 파이썬에서 조건값을 읽고 어느 함수를 부를지 정하기 때문입니다. 트레이싱 단계에서 그래프가 쪼개지지 않는다고 해서 host 개입 없이 재생할 수 있는 것은 아니었습니다.

다음으로 예상한 대로 조건부 레이어 위치에 GPU 유휴 구간이 생겼습니다. gather_scatter는 계산량이 masked의 절반인데도 scalar `if`와 거의 같은 크기의 유휴 구간을 만들었습니다. `nonzero()`가 결과 길이를 알기 위해 host와 두 번 동기화하기 때문입니다. 계산량을 줄여도 host 동기화는 그대로 남습니다. 덧붙이면 처음에는 조건을 읽는 코드를 레이어 루프 밖에 둬서 동기화가 파이프라인 시작 전에 일어났습니다. 16번 레이어 위치로 옮겨 다시 측정한 결과가 위 표입니다.

마지막으로 eager 실행 시간 차이는 5% 남짓이었습니다. 한 경로만 계산하는 이득이 나머지 31개 레이어에 묻혔습니다. 이 규모에서는 실행 시간보다 캡처가 되느냐 같은 구조 차이가 더 중요했습니다. CPU에서는 그래프가 하나로 묶여도 `torch.compile` 버전이 eager보다 오히려 느렸습니다(0.86~0.89배). 그래프 구조의 차이는 GPU에서 의미가 있습니다.

### 6. host를 거치지 않을 수 있을까

CUDA 12.3부터는 조건을 device에서 평가하는 그래프 노드(`cudaGraphConditionalHandle`, `cudaGraphSetConditional`)를 쓸 수 있습니다. Hopper 전용으로 소개되는 경우가 많아 A10G(Ampere)에서 CUDA C++로 직접 짜 보니 정상 동작했습니다. 같은 작업을 host가 조건을 읽어 분기하는 방식과 비교했습니다.

| 규모 | device-side 조건 노드 | host 왕복 | 차이 |
|---|---|---|---|
| 단일 연산 (2,000회) | **0.00986 ms** | 0.01521 ms | device-side가 1.54배 빠름 |
| 32-layer 파이프라인 (200회) | **8.46058 ms** | 8.48112 ms | 0.24% 차이로 거의 같음 |

단일 연산에서 1.54배였던 차이는 32-layer 파이프라인에서 0.24%로 거의 사라졌습니다. 파이프라인 버전은 cuBLAS 대신 직접 짠 단순 GEMM 커널(hidden_dim=128)로 만들었습니다. 이 커널이 레이어당 약 264μs씩 걸리다 보니 host 왕복 비용(약 20μs)이 그 안에 묻혔습니다.

host 왕복 비용은 측정 방식에 따라 수 μs에서 수십 μs 사이였습니다. 이 비용이 전체에서 차지하는 비중은 주변 계산량에 따라 달라집니다. 5절의 cuBLAS 파이프라인에서 47μs짜리 유휴 구간은 전체의 약 1.3%였습니다. 커널 하나하나가 더 짧은 작은 배치 decode라면 그 비중은 더 커질 것으로 보입니다.

### 7. 마치며

분기를 조건별 그래프 두 벌로 나눌 수 있다면 그쪽이 여전히 가장 좋은 선택입니다. 아래 기준은 루프나 분기 개수 때문에 그렇게 할 수 없을 때를 위한 것입니다.

- **조건이 행마다 다르면** masked를 기본값으로 둡니다. 계산이 두 배로 드는 손해는 분명하지만 그래프가 고정되고 CUDA Graph로 캡처됩니다. gather_scatter는 branch 계산이 크고 그래프 분할과 데이터 의존 크기를 감당할 수 있을 때 고려합니다.
- **조건이 배치 단위면** 진짜 분기가 줄어든 계산량만큼 빠릅니다. 다만 쓰려는 런타임에서 직접 측정해 봐야 하고(ONNX Runtime이 반례였습니다), 호출마다 조건값을 host로 읽어 오는 비용이 생긴다는 점도 감안해야 합니다.
- **반복 파이프라인 안이라면** FLOPs보다 먼저 host 동기화가 생기는지, CUDA Graph로 캡처되는지를 봅니다. 커널이 짧은 작은 배치일수록 그 비중이 커집니다.

이번에 실험하지 못한 부분도 있습니다. 배치 크기를 바꿔 가며 masked와 gather_scatter의 처리량이 역전되는 지점은 직접 측정하지 않았습니다. device-side 조건 노드를 cuBLAS 기반 파이프라인에 붙여 보는 일과 Hopper에서 같은 실험을 반복하는 일도 다음 과제로 남겨 둡니다. 분기가 든 모델을 엔진으로 옮길 일이 있다면, 속도를 측정하기 전에 nsys 타임라인에서 그 레이어 자리에 빈칸이 생기는지부터 확인해 보시길 권합니다.

---

### 실험 환경

- CPU: Apple M2 Max, PyTorch 2.13, ONNX Runtime 1.27, JAX 0.11
- GPU: AWS g5.xlarge (NVIDIA A10G), CUDA 12.8, TensorRT 11.1, PyTorch 2.6

### 스펙 확인에 쓴 문서

- [Working with Conditionals (NVIDIA TensorRT)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-conditionals.html)
- [If (ONNX operator spec)](https://onnx.ai/onnx/operators/onnx__If.html)
- [jax.lax.cond (JAX documentation)](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
- [Control Flow - Cond (PyTorch documentation)](https://docs.pytorch.org/docs/2.12/higher_order_ops/cond.html)
- [NonZero (NVIDIA TensorRT Operators)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/operators/docs/NonZero.html)
- [CUDA Graphs, conditional nodes (CUDA Programming Guide)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

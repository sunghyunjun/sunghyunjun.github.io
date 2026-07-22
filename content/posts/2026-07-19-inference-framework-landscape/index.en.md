---
title: "Inference Optimization: What Should You Choose?"
date: 2026-07-19
categories: [Engineering]
tags: [Inference, Optimization, PyTorch, LLM, Compiler]
draft: false
---

### 1. An Overview of Inference Optimization Techniques

Running a model as a production service requires an inference optimization step. Reaching for a proven framework or library is the natural approach. Rather than writing kernels from scratch, it's safer and more efficient to build on compilers and runtimes that other teams have already validated.

The problem is that this list of options changes fast. Up through 2024, TorchScript was the obvious starting point whenever inference optimization came up in the PyTorch world. By mid-2026, that landscape has shifted considerably (more on this in Section 4). It seems worth taking stock of what the options actually look like right now.

This post divides today's inference-optimization tools into three broad categories: general-purpose graph compilers/runtimes, LLM-specific serving engines, and runtimes specialized for mobile/edge. From there it covers changes to PyTorch's own deployment paths and vendor-specific platform options, and closes with a coverage matrix across five device axes: NVIDIA, AMD, Apple Silicon, CPU, and mobile.

{{< svg name="diagram-1-stack.en.svg" caption="Figure 1. Inference-optimization tools split into three branches, sitting on a shared hardware foundation." >}}

### 2. Compilers and Graph Runtimes

Some tools treat a model as a single computation graph and compile it into a form optimized for the target hardware. TensorRT [[4]](https://github.com/NVIDIA/TensorRT/releases) [[5]](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html), ONNX Runtime [[1]](https://github.com/microsoft/onnxruntime/releases), OpenVINO [[2]](https://github.com/openvinotoolkit/openvino/releases), Apache TVM [[6]](https://github.com/apache/tvm/releases), the MLIR-based IREE [[7]](https://github.com/iree-org/iree), and StableHLO [[8]](https://github.com/openxla/stablehlo) — the portability layer connecting XLA and IREE — all fall into this category. (I haven't worked with IREE or StableHLO directly myself, so this section sticks mostly to facts confirmed in their official docs.)

As of mid-2026, all of these tools are under active maintenance. ONNX Runtime ships releases almost every month [[1]](https://github.com/microsoft/onnxruntime/releases). OpenVINO keeps a 6–8 week cadence and recently added a preview backend for llama.cpp, a path that runs GGUF models directly on Intel CPU/GPU/NPU [[2]](https://github.com/openvinotoolkit/openvino/releases) [[3]](https://www.phoronix.com/news/OpenVINO-2026.1-Released).

TensorRT keeps its own separate release cycle going [[4]](https://github.com/NVIDIA/TensorRT/releases) [[5]](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html). Apache TVM is also under continued development, with dozens of PRs landing per release [[6]](https://github.com/apache/tvm/releases) — though the recent changelog is dominated by CUDA-related entries with almost no mention of ROCm or Apple, which gives some sense of where the engineering effort is concentrated. IREE supports CUDA, ROCm, Metal, Vulkan, and CPU per its documentation [[7]](https://github.com/iree-org/iree). It's still a sandbox-stage project under the LF AI & Data Foundation, which is worth keeping in mind.

What these tools have in common is that the graph itself doesn't change the shape of the next computation based on the previous output. That doesn't mean the graph is called exactly once — a diffusion model, for example, calls the same graph repeatedly across many steps, and the input shape can be either static or dynamic. But that repetition is driven by an orchestration layer calling the graph a fixed number of times; the model itself doesn't decide during execution how much more computation is left to do. This is exactly where the LLM decode loop differs.

### 3. General Neural Networks vs. LLMs

The compilers covered in Section 2 rest on the premise above: the graph doesn't dynamically decide its own next computation. That premise holds up well for image classification or speech recognition, where the amount of compute per request doesn't vary much.

LLM inference breaks that premise in two ways.

First, LLMs generate tokens one at a time in an autoregressive decoding structure. The prefill stage, which processes the prompt in one shot, is compute-bound. The decode stage that follows, pulling out one token at a time, is memory-bound — the compute itself is small, but memory access dominates. Each decode step also grows the KV cache, which holds the attention state for every previous token, in proportion to sequence length. The graph's own state keeps growing while it runs, which is a different problem from what the static compiled graphs in Section 2 assume.

Second, production serving doesn't have requests arriving or finishing all at once. Arrival time, prompt length, and output length all vary from request to request. That's why serving needs continuous batching: rather than fixing a batch, it keeps refilling the GPU as requests come and go.

{{< svg name="diagram-2-general-vs-llm.en.svg" caption="Figure 2. A general neural network's forward pass vs. an LLM's autoregressive decoding, growing KV cache, and continuous batching." >}}

vLLM, SGLang, TensorRT-LLM, and llama.cpp — the LLM-specific serving engines — exist to handle exactly these two things: autoregressive decoding and continuous batching.

Per its own docs, vLLM supports not just NVIDIA CUDA but also AMD ROCm, Google TPU, and Intel Gaudi/XPU [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). Apple Silicon isn't in the main installation guide and is only reachable through the third-party vLLM-Metal path [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). The 2026 Q2 AMD roadmap shows the new V1 engine's ROCm port is still in progress [[10]](https://github.com/vllm-project/vllm/issues/44092).

TensorRT-LLM shows a clear tradeoff typical of compiled engines. According to one benchmark, cold-start/build time is far longer than vLLM or SGLang (~28 minutes vs. roughly a minute). In exchange, TRT-LLM's peak throughput sometimes comes out ahead in high-concurrency scenarios on H100 [[14]](https://leetllm.com/blog/llm-inference-engine-comparison-2026).

SGLang's Apple Silicon support started from zero as of its February 2026 roadmap. There's an ongoing experiment to swap out PyTorch for MLX, but as of July that same year, the core serving kernels are still unimplemented [[13]](https://github.com/sgl-project/sglang/issues/19137). llama.cpp/GGML, by contrast, moves fast enough to ship releases daily, with its own implementations covering essentially every vendor backend — CUDA, ROCm, Metal, Vulkan, SYCL [[15]](https://github.com/ggml-org/llama.cpp/releases). Between OpenVINO's llama.cpp backend and Ollama's reliance on llama.cpp [[3]](https://www.phoronix.com/news/OpenVINO-2026.1-Released) [[14]](https://leetllm.com/blog/llm-inference-engine-comparison-2026), GGUF also looks like it's settling in as a cross-engine exchange format.

> **Disaggregated serving.** There's also an orchestration layer that sits on top of the serving engines. NVIDIA Dynamo isn't a successor to TensorRT-LLM — it's a distributed serving layer that can swap in PyTorch, SGLang, TensorRT-LLM, or vLLM as a backend [[11]](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/). The core idea is to physically split compute-bound prefill and memory-bound decode across different GPUs/nodes so each can scale independently. That's disaggregated serving. And NVIDIA Triton Inference Server hasn't gone away — it's still being actively promoted, just renamed "NVIDIA Dynamo-Triton" [[12]](https://developer.nvidia.com/triton-inference-server). General-purpose inference is handled by Dynamo-Triton, while LLM-specific optimization is handled by the separate Dynamo.

### 4. Changes in the PyTorch Ecosystem

It seems like using TorchScript for inference deployment was pretty much the default in the PyTorch world for a long time.

In February 2024, a PyTorch core developer stated on the dev-discuss forum that "TorchScript won't be deprecated without a technically superior replacement" [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642). At that point, AOTInductor was still a prototype [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642).

Two years later, PyTorch 2.10 officially deprecated TorchScript. The 2.11 release blog states plainly that "torch.export replaces the jit trace/script APIs, and ExecuTorch replaces the embedded runtime" [[17]](https://pytorch.org/blog/pytorch-2-11-release-blog/). The GitHub release notes even note that torch.jit isn't guaranteed to work on Python 3.14 [[19]](https://github.com/pytorch/pytorch/releases). In the official docs, the "TorchScript Unsupported PyTorch Constructs" page was removed and now redirects to the torch.compiler API reference [[18]](https://docs.pytorch.org/docs/stable/jit_unsupported.html).

{{< svg name="diagram-3-pytorch-timeline.en.svg" caption="Figure 3. TorchScript's decline and the rise of torch.export, AOTInductor, ExecuTorch, and torch.compile." >}}

Three paths have largely split up what TorchScript used to cover.

**torch.export** produces a graph that's fully decoupled from the Python runtime, so it can be loaded and run in other environments and languages [[20]](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html). It explicitly claims a soundness property `torch.jit.trace()` never had: it tracks integer computations on sizes and guarantees the trace stays valid for other inputs.

**AOTInductor** is a specialized version of TorchInductor that takes only torch.export output as input, compiling it into a shared library (`.so`) for deployment in non-Python production environments like C++ [[21]](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). It isn't portable across devices, though — a `.so` compiled for CUDA can't run on CPU [[16]](https://dev-discuss.pytorch.org/t/whats-the-difference-between-torch-export-torchserve-executorch-aotinductor/1642).

**ExecuTorch** was officially designated the successor to the "embedded runtime," but its scope has grown well beyond that. Per its docs it supports 14 backends — XNNPACK (CPU everywhere), Metal/Core ML (Apple), Vulkan, Qualcomm, MediaTek, Arm, and more — including a CUDA backend for Linux/Windows [[22]](https://docs.pytorch.org/executorch/stable/backends-overview.html). There's no AMD ROCm backend yet.

**torch.compile/Inductor** keeps expanding independently of all this. Inductor uses OpenAI Triton as its default code-generation backend, and 2.13.0 added a second, CuTeDSL-based code-generation path as a prototype. The same release ported FlexAttention to Apple Silicon (MPS), showing up to a 12x speedup over SDPA on sparse patterns, while ROCm gained AOTriton 0.12b and Arm gained Armv9-A targeting [[19]](https://github.com/pytorch/pytorch/releases).

Worth noting here: torch.compile's scope isn't limited to inference. Because it captures both the forward and backward graphs together via AOTAutograd, the backpropagation step during training gets optimized through the same compilation path. This post looks at things from an inference angle, but PyTorch 2.x's compilation stack itself is a single architecture that doesn't separate inference from training.

So the gap left by TorchScript wasn't filled by a single successor — three paths split it between them: the server (torch.export + AOTInductor), embedded (ExecuTorch), and eager-mode optimization (torch.compile).

### 5. Vendor and Platform-Specific Options

So far this has covered cross-vendor, general-purpose tools. This section turns to vendor-specific stacks that try to cover the full CPU/GPU/NPU stack within a single platform.

Apple has two complementary options: **Core ML** and **MLX**. Core ML deploys across the entire iOS/macOS/watchOS/tvOS lineup and gets direct access to the Neural Engine, making it the production-deployment option. MLX only runs on Apple Silicon Macs and can't be deployed to mobile, but it's built for research and fine-tuning instead [[23]](https://cactuscompute.com/compare/coreml-vs-mlx). Apple's own published M5 benchmarks make this split more concrete: the GPU Neural Accelerator sped up MLX's large matrix multiplications (compute-bound, TTFT-heavy) by up to 4x, but only improved memory-bound token generation by 1.2–1.3x [[24]](https://machinelearning.apple.com/research/exploring-llms-mlx-m5). That's a good illustration of how the same chip generation can deliver very different gains depending on the workload.

On the AMD side, **MIGraphX** is a graph compiler that takes ONNX/TensorFlow models and optimizes them for AMD GPUs (via MIOpen/rocBLAS) and CPUs (via DNNL/ZenDNN) [[25]](https://rocm.docs.amd.com/en/docs-6.0.0/conceptual/ai-migraphx-optimization.html). For a more current read on the vitality of the ROCm ecosystem, vLLM's AMD roadmap (Section 3) is a more recent signal than this documentation.

For mobile and edge, **LiteRT** and **MNN** are the standouts. LiteRT is Google's on-device runtime, renamed from TensorFlow Lite in 2024 [[26]](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/). It's moved past TensorFlow-only support to accept PyTorch and JAX models too, and it abstracts NPUs from Google Tensor, Intel, MediaTek, and Qualcomm behind a single Compiled Model API [[27]](https://github.com/google-ai-edge/litert). It has no CUDA/ROCm desktop GPU backend, staying firmly focused on the edge. Alibaba's MNN spans CPU, GPU (Metal/Vulkan/CUDA), and NPU (Core ML/HIAI/NNAPI/QNN). MNN-LLM is extending its reach into on-device LLM serving as well [[28]](https://github.com/alibaba/MNN).

One NPU axis that LiteRT, MNN, and ExecuTorch all lean on in common is Qualcomm's Hexagon. **Qualcomm AI Engine Direct** (QNN) bundles Kryo CPU, Adreno GPU, and Hexagon NPU into a single hardware abstraction layer, and frameworks like TFLite and ONNX Runtime reach the Hexagon NPU by delegating to that layer [[29]](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk). A good chunk of the "confirmed" entries in the "mobile/edge" column of the matrix in Section 6 are, under the hood, actually running on Snapdragon through this QNN layer.

At the opposite end sit AWS's own custom silicon, **Trainium and Inferentia**. Both are unified under the AWS Neuron SDK, which is described as letting PyTorch and TensorFlow code carry over largely unchanged [[30]](https://aws.amazon.com/machine-learning/trainium/) [[31]](https://aws.amazon.com/machine-learning/inferentia/). AWS's own materials claim native support for vLLM, HuggingFace Transformers, and TorchTitan [[30]](https://aws.amazon.com/machine-learning/trainium/), but vLLM's own installation docs list Neuron as a third-party plugin rather than part of the main hardware list [[9]](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). It seems fair to say that most of the frameworks covered in this post don't yet reach this silicon — vLLM is about the only confirmed exception right now.

{{< svg name="diagram-4-vendor.en.svg" caption="Figure 4. Apple (Core ML/MLX), AMD (MIGraphX), and mobile/edge (LiteRT/MNN) — three axes each digging deep into their own platform." >}}

### 6. Device Coverage Matrix

Here's everything covered so far organized across five axes: NVIDIA GPU, AMD ROCm, Apple Silicon, general-purpose CPU, and mobile/edge. "Confirmed" means the claim was verified against official docs or release notes. "Not confirmed in this research" doesn't mean the answer is no — it means this particular research pass didn't verify it directly.

{{< svg name="diagram-5-matrix.en.svg" caption="Figure 5. Coverage matrix — 18 frameworks × 5 device axes (as of 2026-07)." >}}

This matrix is a snapshot. Give it six months and a handful of cells will have changed. vLLM's ROCm support might catch up to the V1 engine, and SGLang's Apple backend might grow out of alpha. What matters more than memorizing this matrix is regularly rechecking what's actually verified right now, on the specific device mix your team's workload spans.

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

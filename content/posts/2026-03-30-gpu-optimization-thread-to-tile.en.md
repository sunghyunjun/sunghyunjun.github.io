---
title: "The Shift in GPU Optimization: From Threads to Tiles"
date: 2026-03-30
categories: [Engineering]
tags: [GPU, Optimization, Compiler, MLIR]
draft: false
---

### 1. Introduction: The Fatigue of Model Surgery
In current practice, the most common task for ensuring inference performance when using ONNX Runtime or TensorRT [[1]](https://developer.nvidia.com/tensorrt) is 'Model Rewriting.' This involves replacing operations not supported by inference engines like TensorRT or applying mathematical masking to remove `if` branches that cause Graph Breaks in TorchDynamo [[2]](https://pytorch.org/get-started/pytorch-2.0/). Such tasks can be described as manual interventions by engineers to force a static computation graph and improve compiler optimization efficiency.

### 2. Control Flow and the Future of Model Surgery
This kind of 'surgery' usually occurs when research-stage code is decoupled from the constraints of production environments. Fortunately, as backbone architectures have become standardized with the rise of models like LLMs and Diffusion, the frequency of fragmented model surgery has decreased compared to the past. However, engineers still find themselves repeatedly catering to the whims of the compiler when porting cutting-edge research models to production. As next-generation abstraction technologies take hold, it is expected that the compiler's middle-end will handle these issues more intelligently, rather than requiring engineers to manually rewrite code.

### 3. Transition of Optimization Units: Triton and CUTLASS CuTe
The paradigm of optimization is already shifting from individual thread control to abstractions at the block and tile levels.

*   **Triton** [[3]](https://github.com/triton-lang/triton): Proposes block-level coding to avoid the complexity of loop analysis. Instead of manual indexing at the individual thread level, it lets engineers describe operations directly in block units, delegating GPU memory access pattern optimization to the compiler.
*   **CUTLASS CuTe** [[4]](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quick_start.md): Introduced in CUTLASS 3.0, **CuTe** mathematically abstracts data Layouts and Tiles. Instead of manual indexing dependent on thread hierarchy, it focuses on the **Composability** of mapping data Shapes and Strides to hierarchical hardware resources. This provides an abstraction foundation that allows the same algorithm to be ported to different memory layouts.

### 4. Next-Generation Abstractions: JAX Pallas, Tunix, and CUDA Tile
Recently discussed technologies are pushing these tile-level optimizations into higher levels of the compiler stack.

*   **JAX Pallas** [[5]](https://jax.readthedocs.io/en/latest/pallas/index.html) & **Tunix** [[6]](https://github.com/google/tunix): Google provides a unified tile-centric kernel language through Pallas while simultaneously building MLIR [[7]](https://mlir.llvm.org/)-based compiler infrastructure like **Tunix** behind the scenes. Targeting both TPUs and GPUs, it appears to be pushing hardware-specific optimization decisions up into the compiler level.
*   **CUDA Tile (Tile IR)** [[8]](https://github.com/NVIDIA/cuda-tile): A new programming model released by NVIDIA that moves away from the traditional thread-centric (SIMT) model, elevating **the Tile to a primitive unit of computation**. Rather than requiring engineers to manually rewrite models, it appears to aim for a structure where the compiler determines the optimal tile configuration for next-generation hardware like Blackwell [[9]](https://www.nvidia.com/en-us/data-center/blackwell/).

### 5. Conclusion: Suggestions for Engineering Stance in the 'Auto-Optimization' Era
As technologies like `torch.compile` mature and coding agents (Claude Code, Gemini, etc.) begin to learn optimization patterns and generate kernels, I cautiously predict that the role of engineering will evolve as follows:

*   **Research Engineer**: The need to consider hardware-aware architectures during the model design stage will increase. However, this will move toward utilizing **collaboration with in-house optimization experts, internal knowledge bases, or timely guidance from AI agents** to design mathematical structures of models that are easily understood by optimization tools, rather than requiring individual perfection.
*   **Infrastructure/Enterprise Engineer**: The focus will shift toward managing the efficiency of the entire massive infrastructure rather than manually fixing individual models. This involves designing **compiler policies and standardized optimization passes** that can be applied across thousands of models, effectively creating the 'playground' where AI agents and compilers can perform at their best.
*   **Optimization Specialist**: While rewriting for simple API compatibility will decrease, roles in two core areas are likely to deepen.
    1.  **Deep Surgery**: A highly specialized engineering task involving the redesign of model layouts to fit the latest hardware instructions (Intrinsics) and solving complex data dependencies that compilers cannot yet automate, in order to extract limit performance from the hardware.
    2.  **Defining Primitives**: Defining **entirely new computation primitives for unexplored domains** that have data characteristics completely different from existing AI workloads. The core competency will be the ability to carve out optimization paths and design them into compiler dialects in areas where even coding agents have no reference data.

Ultimately, while the proportion of simple 'surgery' involving manual model modification will decrease, the ability to **orchestrate systems optimized for business** using the compiler stack and AI agents as tools will become even more valuable. We must now begin preparing to move beyond being coders who directly control threads to becoming architects who translate physical hardware characteristics into system-wide optimization strategies.

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

---
title: "The Cost of One Branch in a Compiled Inference Graph"
date: 2026-10-04T00:00:00+09:00
categories: [Engineering]
tags: [Inference, Compiler, CUDA, TensorRT, PyTorch]
draft: false
---

### 1. When a branch has to live inside the graph

Moving a research-stage PyTorch model into a production inference engine usually raises two problems. One is dynamic shape: input lengths and tensor shapes change from request to request. The other is branching inside the model. A Python `if` that takes one line in research code does not carry over to runtimes that fix the graph ahead of time, such as TensorRT or ONNX. When you build an engine or export the model, the code is traced once and becomes a fixed computation graph. The `if` is frozen to the condition value seen during tracing.

The cleanest way to handle a branch is to split the graph by condition. You cut one graph per path, build two engines, and let the host pick which engine to call at run time. Each engine is a static graph with no branch, so it is also easier to optimize.

This breaks down when the branch sits inside a loop or when there are many branches. A branch inside a loop can take a different path on every iteration, so no single engine can be chosen before execution. With several branches, every combination needs its own graph, and the number of engines grows quickly. In those cases the branch itself has to be compiled into the graph.

{{< svg name="fig1-two-ways.en.svg" caption="Figure 1. If the condition is known before execution, split the graph into two engines. If the branch sits inside a loop, the branch itself goes into the graph." >}}

A branch inside the graph comes in two kinds, depending on the unit of the condition. With a scalar condition, the whole batch takes the same path. With a per-row condition, each row in the batch can take a different path. A layer where rows with `route_id` 0 go through `branch_fn_0` and rows with `route_id` 1 go through `branch_fn_1` is an example of the second kind. The conditional operators in these runtimes (TensorRT `IIfConditionalLayer`, ONNX `If`, XLA `Conditional`, `jax.lax.cond`, `torch.cond`) all accept only a scalar condition according to their documentation. Feeding them a per-row condition fails like this:

```
ONNX Runtime   Compute If nodes condition input must have exactly one element
JAX            TypeError: Pred must be a scalar, got ... shape (512,)
torch.compile  Detected data-dependent branching (graph_count=2, one graph break)
```

`torch.compile` still runs, because it splits the graph and stitches the pieces together in eager mode. ONNX and TensorRT deployments, which need to export one whole graph, have no such fallback.

This post compares three ways to put a branch inside the graph, measured on a MacBook CPU and an A10G GPU. It looks at how much computing both paths costs, whether a real branch is faster when the condition is scalar, what happens in the middle of a deep pipeline, and whether a branch can avoid the host altogether. Inside a repeated pipeline, whether the host had to synchronize mattered more than the amount of compute.

### 2. Three approaches

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

{{< svg name="fig2-three-methods.en.svg" caption="Figure 2. Data flow of the three approaches. The labels at the bottom show the strength (blue) and the cost (orange) of each approach." >}}

**masked** computes every row through both paths and multiplies by a mask to keep the result it needs. The condition flows only as numbers in a mask tensor, so the graph has the same shape regardless of the condition. The price is two GEMMs. Running both paths and choosing the result by condition is called predication, so this approach can also be called row-wise predication.

**gather_scatter** uses `nonzero()` to collect the row indices for each path, gathers only those rows (`index_select`), computes them, and writes the results back in place (`index_copy_`). One GEMM is enough. However, the length of the tensor that `nonzero()` returns depends on the values in `route_id`.

**Real branch** (`If`, `torch.cond`) is only available when the whole batch shares one condition. It computes one path, but someone has to read the condition value to decide which path to take.

### 3. How much does computing both paths cost?

I first compared the two per-row approaches on a MacBook (M2 Max) CPU with PyTorch 2.13. Each number is the mean of 30 runs at N=4096 and hidden_dim=1024, and I first checked that all three implementations agree (rtol=1e-5). The split is the fraction of rows with `route_id` equal to 1.

| Implementation | split 0.5 | split 0.01 | vs gather_scatter |
|---|---|---|---|
| **gather_scatter** | **4.62 ms** | **5.05 ms** | fastest |
| masked (eager) | 8.75 ms | 8.81 ms | 1.74 to 1.89× slower |
| masked (`torch.compile`) | 9.02 ms | 9.50 ms | 1.88 to 1.95× slower |

At N=4096, gather_scatter was 1.74 to 1.95 times faster than masked, and about 1.4 times faster at N=512. The split ratio made almost no difference. masked took longer by roughly the cost of a second GEMM. For reference, loop_if, which runs a Python `if` per row, did the least computation but took 118.91 ms, about 14 times slower than masked. The cost of calling an operation from Python for every row outweighed the compute it saved.

When each branch does little work, the order changes. At N=4096 and split 0.5, masked was faster at hidden_dim 32 (0.256 ms against 0.353 ms), and gather_scatter was faster at 128 (0.478 ms against 0.563 ms). gather_scatter launches six extra operations on every call (`nonzero`, `index_select`, and `index_copy_`, two of each), and that overhead is fixed. In the other direction, with 4 and 8 branches masked grew in proportion to the branch count, to 18.78 ms and 37.92 ms, while gather_scatter stayed almost flat at 5.39 ms and 6.90 ms.

{{< svg name="fig3-cpu-cost.en.svg" caption="Figure 3. CPU measurements. (a) masked was faster at hidden_dim 8 and 32, and gather_scatter at 128 and 1024. Panel (a) comes from a separate run that varied hidden_dim, so its ratio at 1024 differs slightly from the table above. (b) As the branch count grows, only masked grows in proportion." >}}

On speed alone, gather_scatter is the better choice. Moving it into a graph causes problems, though. `torch._dynamo.explain` shows the graph breaking at both `nonzero()` calls, which gives graph_count=3. ONNX export succeeds, but the batch axis of the output is declared as a dynamic dimension `n`.

This `n` is different from the dynamic shape in section 1. An input length can be covered by a range registered in an optimization profile ahead of time, while this size is only known after reading tensor values. TensorRT treats such operations as a separate category, Data-Dependent Shape, and a custom implementation needs a much more involved plugin path. For deployment as a static graph, masked remained the safe default.

### 4. What if the whole batch shares one condition?

With a scalar condition, the native conditional operators become available. Since they compute only one path, I expected them to be faster than masked. The settings were N=4096, hidden_dim=1024, and cond=True. CPU numbers are the mean of 50 runs, and A10G numbers the mean of 200.

| Environment | masked (both paths) | real branch (one path) | faster |
|---|---|---|---|
| CPU, Python `if` (eager) | 9.22 ms | **3.95 ms** | real branch, 2.33× |
| CPU, `torch.cond` (compile) | 9.22 ms | **4.23 ms** | real branch, 2.18× |
| CPU, ONNX Runtime `If` | **9.22 ms** | 15.16 ms | masked, 1.64× |
| A10G, TensorRT `If` | 0.693 ms | **0.425 ms** | real branch, 1.63× |

{{< svg name="fig4-scalar-ratio.en.svg" caption="Figure 4. Real branch against masked under a scalar condition. Of the four environments, only ONNX Runtime favored masked." >}}

As expected, the real branch was 2.18 to 2.33 times faster in PyTorch. ONNX Runtime's `If`, however, was the slowest of all even though it computes half as much. The overhead of running a subgraph seems to exceed the savings. When I moved the same pattern to TensorRT on the A10G, `If` was 1.63 times faster. A native `If` was not always faster, and the result depended on the runtime implementation.

One question came up here. To choose the next kernel from a condition computed on the GPU, doesn't someone have to read that value back to the host? Profiling 20 calls with nsys confirmed it. The `If` engine made exactly one 1-byte D2H copy per call, and the masked engine made none. I also checked the GEMM kernel of the masked engine with ncu. Threads within a warp always converged on the same branch target (`branch_targets_threads_uniform` 100%), so execution never diverged inside the kernel. This is because the condition reaches the kernel only as values in the mask tensor.

### 5. Inside a pipeline

So far each measurement was isolated, draining the GPU after every call. In a real model the same layer repeats dozens of times, and the CPU hides launch cost by queuing kernels ahead without synchronizing. If the host has to wait for a condition value in the middle of that flow, does the queue stop there? A branch inside a loop, mentioned in section 1, is exactly this situation.

{{< svg name="fig5-host-sync.en.svg" caption="Figure 5. How host synchronization interrupts asynchronous execution. (a) When the condition flows as data, the CPU queues kernels ahead and the GPU runs without pause. (b) To read the condition on the host, the CPU waits until the earlier kernels and the D2H copy finish. It queues the next kernel only after deciding, which leaves the GPU idle in between. The 46.7 to 47.1 μs value is measured and comes from the table below." >}}

The CPU and the GPU, which were working at the same time, now wait for each other because of one condition value. During that time the GPU receives no work.

To check this, I built a 32-layer pipeline where only layer 16 is conditional (hidden_dim=512, N=2048), in four variants.

| Implementation | Compiled graphs | CUDA Graph capture | Idle gap at layer 16 | D2H copies | One eager run |
|---|---|---|---|---|---|
| masked | 1 | succeeded | none | 0 | 3.48 ms |
| gather_scatter | 3 | failed | 47.1 μs (61× median) | 2 | 3.47 ms |
| scalar `if` | 1* | failed | 46.7 μs (61× median) | 1 | 3.30 ms |
| `torch.cond` | 1 | failed | not measured | not measured | 3.31 ms |

The compiled graph count was measured with `torch._dynamo.explain` on the MacBook CPU, and the other columns on the A10G. *For scalar `if`, explain reports 1, but an actual compile prints a graph break warning at `Tensor.item()`.

{{< figure src="pipeline-timeline-3way.png" caption="Figure 6. Kernel execution on the A10G recorded with nsys. masked runs without gaps, while gather_scatter and scalar if show a GPU idle gap at layer 16." >}}

First, masked was the only variant that CUDA Graph could capture. `torch.cond` failed to capture even though `torch.compile` traced it as a single graph. In eager mode, `torch.cond` still reads the condition in Python to decide which function to call. A graph that does not break during tracing is not necessarily a graph that can replay without the host.

Next, as expected, a real GPU idle gap appeared at the position of the conditional layer. gather_scatter left an idle gap almost as large as scalar `if`, even though it computes half as much as masked. This is because `nonzero()` synchronizes with the host twice to learn the result length. Reducing compute leaves the host synchronization in place. One more note: at first I placed the code that reads the condition outside the layer loop, so the synchronization happened before the pipeline started. The table shows the results after moving it to layer 16 and measuring again.

Finally, the difference in eager run time was only about 5%. The gain from computing one path was diluted across the other 31 layers. At this scale, structural differences such as whether the graph can be captured mattered more than run time. On the CPU, the `torch.compile` version was even slower than eager despite the single graph (0.86 to 0.89×). The structural difference matters on the GPU.

### 6. Can the branch avoid the host?

Since CUDA 12.3, a graph node can evaluate the condition on the device (`cudaGraphConditionalHandle`, `cudaGraphSetConditional`). It is often described as a Hopper feature, so I wrote it directly in CUDA C++ on the A10G (Ampere), and it worked. I compared it with a version where the host reads the condition and branches.

| Scale | device-side conditional node | host round trip | difference |
|---|---|---|---|
| single operation (2,000 runs) | **0.00986 ms** | 0.01521 ms | device-side 1.54× faster |
| 32-layer pipeline (200 runs) | **8.46058 ms** | 8.48112 ms | 0.24%, about the same |

The 1.54 times difference for a single operation almost disappeared, down to 0.24%, in the 32-layer pipeline. The pipeline version used a simple hand-written GEMM kernel (hidden_dim=128) instead of cuBLAS. That kernel took about 264 μs per layer, so the host round trip (about 20 μs) was hidden inside it.

The host round trip cost ranged from a few to a few tens of microseconds, depending on how it was measured. Its share of the total depends on the surrounding compute. In the cuBLAS pipeline from section 5, the 47 μs idle gap was about 1.3% of the total. In small-batch decode, where each kernel is shorter, the share would likely be larger.

### 7. Closing

If a branch can be split into two graphs by condition, that is still the best option. The guidelines below are for cases where a loop or the number of branches rules that out.

- **If the condition differs per row**, use masked as the default. Computing twice is a clear cost, but the graph stays fixed and CUDA Graph can capture it. Consider gather_scatter when each branch is heavy and you can accept a split graph and a data-dependent size.
- **If the condition is per batch**, a real branch is faster by the amount of compute it skips. Measure it on your runtime first (ONNX Runtime was the counterexample), and account for the cost of reading the condition back to the host on every call.
- **Inside a repeated pipeline**, check whether host synchronization occurs and whether CUDA Graph can capture the graph before looking at FLOPs. The smaller the batch and the shorter the kernels, the larger this cost becomes.

Some things I could not test this time. I did not measure the batch size at which the throughput of masked and gather_scatter crosses over. Attaching the device-side conditional node to a cuBLAS-based pipeline and repeating the experiment on Hopper are also left for later. If you need to move a model with a branch into an engine, check the nsys timeline for a gap at that layer before measuring speed.

---

### Environment

- CPU: Apple M2 Max, PyTorch 2.13, ONNX Runtime 1.27, JAX 0.11
- GPU: AWS g5.xlarge (NVIDIA A10G), CUDA 12.8, TensorRT 11.1, PyTorch 2.6

### Documentation used to check the specs

- [Working with Conditionals (NVIDIA TensorRT)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-conditionals.html)
- [If (ONNX operator spec)](https://onnx.ai/onnx/operators/onnx__If.html)
- [jax.lax.cond (JAX documentation)](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
- [Control Flow - Cond (PyTorch documentation)](https://docs.pytorch.org/docs/2.12/higher_order_ops/cond.html)
- [NonZero (NVIDIA TensorRT Operators)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/operators/docs/NonZero.html)
- [CUDA Graphs, conditional nodes (CUDA Programming Guide)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

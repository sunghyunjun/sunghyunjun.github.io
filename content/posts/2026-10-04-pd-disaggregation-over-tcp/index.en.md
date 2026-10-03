---
title: "Chunk First: When Does Prefill–Decode Disaggregation Pay Off over TCP?"
date: 2026-10-04
categories: [Engineering]
tags: [Inference, LLM, vLLM, Serving, Disaggregation]
draft: false
---

Prefill–decode disaggregation (PD disaggregation from here on), which runs prefill and decode on different GPUs, has become one of the most discussed designs in LLM serving. vLLM and SGLang support it as a first-class path, and large services have published production numbers, so it can look like a settled choice. The published results share one premise, though. The nodes are connected by a fast interconnect such as InfiniBand or NVLink, and there are dozens of identical GPUs.

Production infrastructure often looks different. Teams run mixed generations of GPUs, nodes talk over ordinary TCP, and a small service on a few single-GPU cloud instances has no RDMA option at all. For that kind of setup I wanted to know whether PD disaggregation is still worth considering, or whether replicas with a well-tuned chunked prefill are the better answer. I could not find measurements under those conditions, so I made them.

The setup was AWS g6e, g7, and g7e 2xlarge instances (one GPU each, no RDMA) running stock vLLM 0.30.0, with the KV cache moving over NIXL and UCX's TCP path. The models were Gemma 4 26B and 31B, and the study ran 455 runs over nine rounds. This post is a summary of the technical report: questions and answers only. The evidence and the fine print are in the report linked at the end.

### 1. What was compared

Two designs. The replica design runs one vLLM per GPU with chunked prefill, behind a round-robin proxy. The disaggregated design (1P1D) gives prefill to one GPU and decode to another, and the KV cache crosses TCP between them.

Four workloads: 4K-token inputs, 16K-token inputs, ShareGPT chat, and a production trace.

The SLO has two latency conditions. The first token must arrive within 1 s, and after that no two output tokens may be more than 50 ms apart. A design supports a load when at least 90% of requests meet both conditions. The verdict is made per request, because the stall described in the next section does not show up in system-wide percentiles.

### 2. What sets a replica's stall

The prefill chunk size. In a replica, when another request's prefill chunk lands in the same scheduler step as my decode, my tokens stop until that step finishes. So the chunk size bounds the stall.

For the 26B model with 4K inputs, the median longest gap per request was 48 ms at a 2,560-token chunk and 21 ms at 512 tokens.

{{< svg name="fig-chunk-stall.en.svg" caption="Figure 1. Longest gap per request against prefill chunk size. The dotted lines are the 50 ms and 30 ms limits." >}}

This stall is invisible in system-wide p99. In a run where p99 looked normal, 48% of requests had a stall over the limit. That is why the verdict has to be made per request.

### 3. Does disaggregation beat replicas over a slow network

Not against replicas with a tuned chunk. Once the replica chunk was matched to the gap limit (512 to 1,024 tokens for 26B, 256 for 31B), disaggregation over TCP did not serve more load under any latency target at 4K inputs or in chat traffic.

Disaggregation did serve about 3× more in some runs, but only against replicas held at a 2,560-token chunk. vLLM 0.30.0 keeps Gemma 4's chunk at 2,496 tokens or more while video input is enabled, so a stack that pins the chunk is the case this applies to. Where both designs passed, replicas also had the faster first token (78 to 247 ms against 255 to 384 ms), and almost all of the difference was the time spent sending the KV cache.

### 4. Where disaggregation's extra cost comes from

The transfer. UCX's TCP transport defaults to 8 KB segments; raising them to 256 KB alone made the KV transfer 3.4× faster at low load.

After that, things that cannot be configured set the rate. The cloud caps a single flow at about 5 Gbps, 9.5 Gbps inside a cluster placement group, and NIXL splits each transfer across four TCP connections, a number that could not be changed.

The proxies also moved the results. The test proxy from vLLM's NIXL examples paused every stream for 25 to 40 ms during garbage collection, and the proxy in front of the replicas hit aiohttp's default limit of 100 connections, which cost the replicas one load step.

### 5. So when is disaggregation needed

When inputs are long. A longer input needs more prefill chunks, and every chunk is a stall for the other requests on that GPU. Past some length no chunk meets both the gap limit and the TTFT limit, and disaggregation avoids this by moving prefill off the decode GPU.

Above 16K the numbers are a prediction from a model calibrated on the measured runs, not measurements. At low load with a 50 ms gap limit, inputs of roughly 14K to 37K tokens are where 1P1D over TCP meets TTFT limits of a second or more that replicas cannot; the lower edge sits anywhere between 11K and 34K within the model's uncertainty.

{{< svg name="fig-boundary.en.svg" caption="Figure 2. The fastest TTFT each design can reach while keeping the gap under 50 ms, by input length. Above the blue line replicas meet both limits; only in the band between the lines does 1P1D alone meet both. The band exists only from about 14K to 37K tokens." >}}

Three conditions attach. The KV cache per token must be small, which Gemma 4's hybrid attention provides and a model keeping 220 KiB per token does not. The link must be slow; at 95 Gbps the band starts near 5K. And this is a latency result at low load. By throughput, replicas were ahead at every input length.

### 6. What to do first

Treat disaggregation as the last option and check the earlier ones in order.

1. Measure the replicas' longest gap per request at the load you actually serve.
2. Pick the largest chunk that keeps that gap within the limit. If the TTFT at that chunk is within its limit, stop here.
3. Before moving prefill, reduce it: prefix-aware routing, and KV cache loading if prompts repeat.
4. Disaggregate over TCP only when no chunk meets both limits and the transfer time fits the budget. Fix the UCX segment size, the placement group, and the proxy first.

### 7. Closing

Carrying PD disaggregation numbers from fast interconnects into a small TCP setup gives different results. Within the range measured, one conclusion held throughout: tune the chunk before considering disaggregation. The place where disaggregation pays is narrower than expected, long inputs together with a loose TTFT limit.

This is one person's measurement of one model family on three instance types, with no RDMA baseline, and the boundary in Section 5 is a prediction. Read the scope accordingly. The protocol was committed before the first run, and the report records every point where the study departed from it. If you have measured the same question on mixed GPU groups or another model family, I would like to compare notes.

- Technical report: [doi:10.5281/zenodo.23085600](https://doi.org/10.5281/zenodo.23085600)
- Code, configurations, and per-request data: [github.com/sunghyunjun/pd-disaggregation-over-tcp](https://github.com/sunghyunjun/pd-disaggregation-over-tcp), archived at [doi:10.5281/zenodo.23085749](https://doi.org/10.5281/zenodo.23085749)

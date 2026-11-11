---
title: "Chunk First: When Prefill–Decode Disaggregation Pays Off over TCP"
date: 2026-10-04
description: "Technical report, version 1.0. Measurements and a calibrated model on single-GPU cloud instances."
summary: "Technical report, version 1.0. Measurements and a calibrated model on single-GPU cloud instances."
ShowReadingTime: false
ShowWordCount: false
ShowShareButtons: false
ShowPostNavLinks: false
ShowBreadCrumbs: false
citation:
  title: "Chunk First: When Prefill–Decode Disaggregation Pays Off over TCP. Measurements and a Calibrated Model on Single-GPU Cloud Instances"
  authors: ["Jun, Sunghyun"]
  date: "2026/10/04"
  pdf: "papers/chunk-first-v1.0.pdf"
  doi: "10.5281/zenodo.23085600"
  number: "Version 1.0"
  keywords: ["LLM serving", "prefill-decode disaggregation", "chunked prefill", "vLLM", "TCP"]
---

**Sunghyun Jun** · [ORCID 0009-0001-2391-2315](https://orcid.org/0009-0001-2391-2315)

Technical report, version 1.0, October 2026. Measurements and a calibrated model on single-GPU cloud instances.

[PDF](/papers/chunk-first-v1.0.pdf) · [doi:10.5281/zenodo.23085600](https://doi.org/10.5281/zenodo.23085600) · Code and data: [github.com/sunghyunjun/pd-disaggregation-over-tcp](https://github.com/sunghyunjun/pd-disaggregation-over-tcp), archived at [doi:10.5281/zenodo.23085749](https://doi.org/10.5281/zenodo.23085749)

### Abstract

Prefill–decode disaggregation usually relies on RDMA or NVLink, which cheap single-GPU instances lack. For a TCP-linked fleet of such instances, we ask whether disaggregation is worth considering once chunked prefill, which attacks the same decode stalls, is tuned, and if so, where. Serving Gemma 4 26B and 31B with stock vLLM and NIXL/UCX on AWS, we measured both costs: a replica's longest inter-token gap followed the chunk size (26B, 4K inputs: 48 ms at 2,560 tokens, 21 ms at 512), and disaggregation's added time to first token (TTFT) was mostly KV transfer, which a larger UCX segment made 3.4× faster at low load. At 4K inputs and in chat, against tuned chunks, disaggregation served more per GPU under none of the latency targets our measured rates could rank; it won, by about 3× for 26B, only against replicas held at 2,560 tokens, near vLLM v0.30.0's floor for Gemma 4 with video enabled. A step-time model calibrated on our runs predicts that at low load and a 50 ms gap limit, inputs of roughly 14K–37K tokens (lower edge 11K–34K within its uncertainty) need so many chunks that disaggregation over TCP meets TTFT limits of a second or more that no chunk meets, given a small KV cache. Tune the chunk first.

### Cite

```bibtex
@techreport{jun2026chunkfirst,
  author = {Jun, Sunghyun},
  title  = {Chunk First: When Prefill--Decode Disaggregation Pays Off over {TCP}. Measurements and a Calibrated Model on Single-{GPU} Cloud Instances},
  type   = {Technical Report},
  number = {Version 1.0},
  year   = {2026},
  month  = oct,
  doi    = {10.5281/zenodo.23085600}
}
```

A shorter summary is in the blog post [Chunk First: When Does Prefill–Decode Disaggregation Pay Off over TCP?](/en/posts/2026-10-04-pd-disaggregation-over-tcp/)

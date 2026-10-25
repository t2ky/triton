# Triton with Turing (sm_75) tensor cores

This branch is Triton **v3.7.1** with fp16 `tl.dot` lowered to **MMAv2 (`mma.sync`) on sm_75**
(RTX 20xx / T4 / Quadro RTX), instead of the FMA fallback that upstream uses for compute capability < 8.0.

It is a port of [triton-lang/triton#9970](https://github.com/triton-lang/triton/pull/9970) by
[@nlzy](https://github.com/nlzy) ("[BACKEND] Restore MMA support for Turing GPUs"), which was closed
upstream because Turing is not an officially supported target. All credit for the change goes to that PR;
this branch only rebases it onto the release that vLLM 0.31 ships with.

## Why

Since Triton 3.3 ([triton-lang/triton#5066](https://github.com/triton-lang/triton/pull/5066)),
`AccelerateMatmul` rejects MMA for compute capability < 80 and falls back to FMA. Results stay correct,
but every Triton-based kernel on Turing runs without tensor cores. In vLLM this hits the Triton attention
backends (`TRITON_ATTN`, `FLEX_ATTENTION`) and the multi-LoRA (punica) shrink/expand kernels.
See also [vllm-project/vllm#60821](https://github.com/vllm-project/vllm/issues/60821).

## What changes

Three files, +28/-16 (identical in intent to #9970):

- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` — allow MMA from sm_75; on sm_75 only for
  fp16 x fp16 -> fp16/fp32 (other element types keep the FMA path).
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp` — pass `batchOffset` to the Turing
  fp16/int8 MMA helpers so batched (3D) dots work; drop the "batch size 1 only" assert.
- `test/TritonGPU/accelerate-matmul.mlir` — tests updated accordingly.

## Results (RTX 2080, sm_75, CUDA 13.0, driver 580)

PTX of a 64x64x32 fp16 `tl.dot` compiled for `cuda:75`:

| | `mma.sync` | `fma.rn.f32` |
|---|---:|---:|
| upstream 3.7.1 | 0 | 1024 |
| this branch | 32 | 0 |

A simple tiled fp16 matmul (fp32 accumulate, BLOCK 64x64x32):

| M x N x K | upstream 3.7.1 | this branch | max rel. error vs torch |
|---|---:|---:|---:|
| 256 x 512 x 1024 | 2.1 TFLOP/s | 7.8 TFLOP/s | 2.6e-06 |
| 1024 x 1024 x 4096 | 1.7 TFLOP/s | **22.5 TFLOP/s** | 1.1e-05 |

End to end with vLLM 0.31.0 (Gemma 4 12B IT, GPTQ 4-bit, TP=2 on 2x RTX 2080 8GB, `TRITON_ATTN`,
one LoRA adapter rank 16 enabled, ~800-token prompts):

| | upstream 3.7.1 | this branch |
|---|---:|---:|
| decode time per token (1 stream) | 50.9 ms | **18.6 ms** |
| median latency, 8 concurrent | 13.1 s | **4.7 s** |
| aggregate generation, 8 concurrent | 35 tok/s | **93 tok/s** |

Greedy outputs matched a reference run on an RTX 3090 (stock vLLM, TP=1) exactly for 17 of 20 prompts;
the other 3 diverged late into equally fluent text (expected from fp16 rounding with a different TP split).

Note: Gemma 4's 512-wide global attention also needs a vLLM-side tile fix to fit Turing's 64 KB shared
memory (see [vllm-project/vllm#39018](https://github.com/vllm-project/vllm/pull/39018)); that is not part of Triton.

## Build

Build the wheel inside the image you will run it in, so the Python ABI matches (vLLM 0.31.0 = Python 3.12):

```bash
docker run --rm -v $PWD:/src -v $PWD/dist:/dist -e MAX_JOBS=12 -e TRITON_WHEEL_VERSION_SUFFIX=+turing \
  --entrypoint bash vllm/vllm-openai:v0.31.0 -c '
    apt-get update -qq && apt-get install -y -qq build-essential git zlib1g-dev
    pip install -r /src/python/requirements.txt
    cd /src && git config --global --add safe.directory /src
    pip wheel --no-build-isolation --no-deps -w /dist .'
# then, in your vLLM image:
pip install --no-deps --force-reinstall dist/triton-3.7.1+*.turing-cp312-*.whl
```

About 20 minutes on 16 cores; the LLVM build is downloaded prebuilt.

## Limitations

- Only fp16 operands use MMA on sm_75. bf16 has no tensor-core support on Turing; int8 and fp8 keep the FMA path.
- Turing has no `cp.async`, so software pipelining (`num_stages > 1`) does not overlap loads the way it does on sm_80+.
- Not covered by upstream CI. Run your own correctness checks (the numbers above came from comparing against `torch.matmul` and against a reference GPU).

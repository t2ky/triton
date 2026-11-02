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
- Turing has no `cp.async`, so by default software pipelining (`num_stages > 1`) is skipped on sm_75. See the
  experimental register-staged pipeline below.
- Not covered by upstream CI. Run your own correctness checks (the numbers above came from comparing against `torch.matmul` and against a reference GPU).

## Experimental: register-staged software pipelining on sm_75 (`TRITON_TURING_PIPELINE=1`, off by default)

Upstream runs the pipeliner (`assign_latencies` / `schedule_loops` / `pipeline`) only on sm_8x/9x because it
relies on `cp.async`. This branch adds an opt-in path for sm_75 that stages loads through registers, the same
shape CUTLASS uses on sm75:

```
[tt.load of iteration k+1 -> registers] [local_load buffer k + dot] [local_store registers -> buffer k+1]
```

- `TRITON_TURING_PIPELINE=1` enables the pipeliner passes on sm_7x (`hopper_warpspec` stays sm_80+ only).
  `TRITON_NV_REG_PIPELINE=1` forces the same register-staged lowering on sm_80+ (for development on newer GPUs).
  Both are in `CACHE_INVALIDATING_ENV_VARS`, so switching them recompiles instead of reusing cached kernels.
- `LowerLoops`: `createRegStagedCopy` keeps the `tt.load`, replaces its users with `local_load` of the current
  buffer and stores into the next buffer in the last cluster (2+ buffers). A "producer staging" variant handles
  `tt.load -> pure compute (e.g. int4 -> fp16 dequant) -> local_alloc -> dot` by moving the compute next to the
  store, so dequantized weights do not take an extra round trip through shared memory.
- `Prefetch`: loops built this way carry "the buffer to read this iteration" as an iter_arg, so the existing
  prefetch pass splits the dot along k and loads the first k-slice of the next iteration early.
- `TritonGPUToLLVM`: skips the read-after-write barrier between the staged `local_load` (buffer k) and
  `local_store` (buffer k+1), which are always different buffers. One `__syncthreads` per iteration remains.
- Loops with two or more dots (attention) and loops that cannot use register staging are left unpipelined on
  sm_75 (pipelining them would emit `cp.async`). Attention got slower with pipelining in our tests.

Measured on RTX 2080 (fp16 GEMM, M=2048, TFLOP/s; cuBLAS 53-61): best tiles without pipelining reach
about 44-51; with this path, e.g. a 3840x15360 GEMM goes 44.7 -> 52.4 (128x128x32, 4 warps, 2 stages).
For a dense fp16 layer stack (qkv + o + gate_up + down) the total time drops 12-15% at M=1024-2048.
4-bit weight GEMMs (dequant in the loop) gain 18-40% over the same tile without pipelining but do not
beat Marlin. Correctness: `python/test/unit/language/test_core.py` with `TRITON_NV_REG_PIPELINE=1` on an
RTX 3090 shows no numerical failures (3 failures: two tests that assert `cp.async` waits are emitted and one
shared-memory-size failure that also happens without the flag).

Gains are kernel-dependent and end-to-end serving gains were small in our tests, so this stays opt-in.

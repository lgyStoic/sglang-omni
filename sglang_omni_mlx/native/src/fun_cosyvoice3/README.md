# CosyVoice3 native LLM probe

Work in progress for #2888. This is a numeric-ID LLM probe, not a complete TTS implementation or server. Flow/DiT, HiFT, reference-audio preprocessing, tokenization, production sampling and streaming are not implemented here. An optional bounded greedy mode is a diagnostic only. Quantized checkpoint loading is implemented but not yet runtime-validated.

## Reference and scope

The reference is `sglang_omni/models/fun_cosyvoice3/mlx/model.py` at `28c3e35a33bef2f45b3315431f4a512d6ed463cf`, immediately before the legacy path was removed. The parity script verifies the reference file SHA-256 before executing it. Retrieve that file from Git history and supply it explicitly; the script does not download or execute an unpinned remote module.

The probe reuses the existing decoder with Q/K normalization disabled and half-split RoPE, loads the fine-tuned Qwen2 weights, builds `[SOS, text, TASK, reference speech]`, and projects the final normalized hidden state through the separate speech head. Text IDs must already include any reference transcript and tokenizer control tokens. Forced decode IDs exercise the KV cache without sampling.

Build the `cosyvoice3_llm_probe` CMake target using MLX 0.32.3. Run `cosyvoice3_llm_probe --help` for the standalone CLI. Its forced-decode logits output is a float32 NPY array shaped `[1 + decode_token_count, 6761]`. Optional `--greedy-max-tokens N --greedy-min-tokens N` instead exports one row per selection (including the terminal selection), with IDs and stop reason in stdout JSON. Greedy is exclusive with `--decode-ids`, bounded to 256 selections, selects the lowest ID on ties, masks controls until the minimum, and stops on any control ID ≥6561. It applies no repetition penalty or other production sampling policy.

## Reproduce the comparison

From the repository root, with MLX 0.32.3, mlx-lm 0.32.2, PyTorch and NumPy available:

```sh
python sglang_omni_mlx/native/ci/cosyvoice3_llm_parity.py \
  --reference-file /path/to/pinned/model.py \
  --raw-model /path/to/official/model-bundle \
  --probe /path/to/build/cosyvoice3_llm_probe \
  --output-dir /path/to/local-validation-output
```

The official bundle must contain `llm.pt` and `CosyVoice-BlankEN/config.json`. The script exports an FP16 converted checkpoint using the legacy loader, records source and output SHA-256s and package versions, and compares prompt embeddings and prefill/decode logits. Use a local filesystem: the existing shared safetensors loader also attempts to load macOS AppleDouble files such as `._model.safetensors` on some network volumes.

## Initial measured results

With Python 3.12.15, MLX 0.32.3, mlx-lm 0.32.2, PyTorch 2.14.1 and NumPy 2.3.5 on Apple Silicon:

- Two fixed numeric-ID prompts, text-only and text plus reference speech, each with two forced decode steps: native/reference embeddings and every logits row matched exactly (maximum absolute difference 0).
- Six malformed/out-of-range input cases were rejected as expected.
- The full harness **fails** its separate cached-versus-full-prefix check for the text-only prompt at `atol=0.02`, `rtol=0.002`. Both Python and native have maximum absolute difference 0.0302734375, with matching argmax. The text-plus-speech case passes (maximum difference 0.01171875). Tolerances have not been relaxed.

These small fixtures establish initial implementation parity only, not speech quality, model-wide correctness, quantized correctness, throughput, or end-to-end TTS support. Investigate FP16 cache/full-prefix differences before treating the full harness as a passing regression gate. Mac CI goldens and full model pins are still pending.

## PyTorch component and greedy diagnostics

The additional tools use Transformers `Qwen2Model`, not Python MLX or the full SGLang serving runner. Both independently verify every converted tensor against `llm.pt` rounded to FP16, allow only the unused text `llm.model.lm_head.weight` to be omitted, strictly load the fine-tuned backbone, and verify loaded parameters, speech embeddings and speech head. FP32 component modes widen these same rounded weights; they do not compare native FP32 or unrounded raw weights. There is no dependency on an earlier report, legacy module, or the current CUDA path.

Run as modules from the repository root with PyTorch, Transformers, safetensors and NumPy installed alongside the MLX build environment (measured with Transformers 5.19.0). Run without `-O`, `-OO` or `PYTHONOPTIMIZE`: both CLIs reject optimized execution before creating outputs or loading models because their validation requires assertions. Direct calls to the shared weight-verification and metric helpers enforce the same restriction. Use a **new output directory** on each run; the tools refuse to overwrite previous results. The converted checkpoint must be on a local filesystem. For MPS, explicitly disable fallback before starting Python; an unavailable MPS backend or enabled fallback is an error, never an implicit CPU retry.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 MLX_ENABLE_TF32=0 \
python -m sglang_omni_mlx.native.ci.cosyvoice3_torch_parity \
  --raw-model /path/to/official/model-bundle \
  --converted /path/to/converted \
  --probe /path/to/build/cosyvoice3_llm_probe \
  --backend all --output /path/to/new-component-output

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 MLX_ENABLE_TF32=0 \
python -m sglang_omni_mlx.native.ci.cosyvoice3_greedy_parity \
  --raw-model /path/to/official/model-bundle \
  --converted /path/to/converted \
  --probe /path/to/build/cosyvoice3_llm_probe \
  --backend mps --output /path/to/new-greedy-output
```

Component `--backend all` runs CPU FP32 eager, MPS FP16 eager, MPS FP32 eager and MPS FP16 SDPA; `cpu` or `mps` explicitly selects that subset. Six synthetic numeric fixtures cover prefill plus two or four forced decode steps, both cached and full-prefix, up to 198 prefix tokens. Greedy accepts `cpu` or `mps` and uses FP16 SDPA; only its MPS path has been measured here. Native always uses MLX, including when the Torch reference is CPU.

Greedy pins the actual separately encoded prefix and target IDs for four natural English/Chinese texts in the script, including `You are a helpful assistant.<|endofprompt|>` with end-of-prompt ID 151646. These are the original CosyVoice-BlankEN tokenizations, not generated numeric stress inputs. This deliberately tests fixed token inputs rather than re-running a tokenizer or extracting special tokens from another runtime's source. It does **not** validate tokenizer/frontend behavior. Both backends generate independently, using a minimum of twice the target token count and maximum `min(256, 20 * target_token_count)`; no reference speech, repetition penalty, RAS, or silent-token suppression is applied.

For tokenizer-only reproduction, use the official `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` bundle's `CosyVoice-BlankEN` directory. The following artifact SHA-256s were rechecked against the original local `greedy-generation/report.json` (Transformers 5.19.0, tokenizers 0.23.3):

| Tokenizer artifact | SHA-256 |
|---|---|
| `tokenizer_config.json` | `482bd979881423375ca5414e4e0d94cd7c5349dbb17fffd46b4d36d71e62a1bc` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | `ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0` |

This reproduction registers only the special-token prefix needed by these fixtures; the full service also registers later audio/phoneme tokens, which none of these texts use. It performs no inference and imports no service/CUDA utilities. Check the three files with `shasum -a 256` against the table, then run from the repository root without Python optimization:

```sh
python - /path/to/official/model-bundle/CosyVoice-BlankEN <<'PY'
import sys
from transformers import AutoTokenizer
from sglang_omni_mlx.native.ci.cosyvoice3_greedy_parity import (
    PREFIX, PREFIX_IDS, TEXT_FIXTURES,
)

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|endofprompt|>"],
})
assert tokenizer([PREFIX], return_tensors="pt")["input_ids"][0].tolist() == PREFIX_IDS
for name, text, expected_ids in TEXT_FIXTURES:
    actual_ids = tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()
    assert actual_ids == expected_ids, (name, actual_ids, expected_ids)
print("Prefix and all four target tokenizations match")
PY
```

Reports include package versions, input IDs, checkpoint/config/binary hashes, backend settings, weight-equivalence checks, finite-logit checks and numerical metrics. Greedy records both complete sequences, stop reasons and the first divergence; numerical comparisons stop after the first divergent selection so later mismatched contexts are not compared. `status: completed` means execution finished, not parity passed. `numerical_pass` uses the unchanged diagnostic threshold `abs(native - torch) <= 0.02 + 0.002 * abs(torch)`; top-1/sequence agreement is reported separately. Exit 1 means a completed numerical or sequence comparison failed (execution errors also return nonzero; inspect the report/traceback). A length cap is not successful natural termination. These diagnostics do not replace or weaken the original MLX gate.

### Measured outcomes after cleanup

On the same Apple Silicon stack, all 292 converted tensors and all prompt embeddings verified exactly. Every logit was finite. All four component modes completed; each failed the numerical threshold on some rows despite top-1 agreement on all 52 cached/full rows per mode. Maximum absolute errors were 0.192423 (CPU FP32 eager), 0.150391 (MPS FP16 eager), 0.192286 (MPS FP32 eager), and 0.136475 (MPS FP16 SDPA), each against native FP16.

| Greedy text | Selections on each backend | Stop on both |
|---|---:|---|
| Hello, how are you? | 120 | Generation cap, repetitive tail |
| The sun is shining today. | 120 | Generation cap, repetitive tail |
| 你好，很高兴见到你。 | 200 | Generation cap, repetitive tail |
| 今天天气很好，我们去散步吧。 | 68 | EOS 6562 |

All 508 selections agreed, but three of four cases hit the cap on both backends and repetitions reduce decision diversity. There were still 39 logits outside the numerical threshold. The original unmodified MLX harness also still exits 1: its text-only cached/full-prefix maximum difference remains 0.0302734375, while native/legacy matched-style comparisons and all six invalid-input checks pass. No thresholds or gates were relaxed.

This is limited component/trajectory evidence on one software/hardware stack, not full MPS serving parity, production-policy termination, audio quality, quantized validation or a performance claim. The original golden coverage is only two small fixtures. Precision-control experiments suggest low-precision arithmetic differences, but do not establish a specific kernel defect; those bulk traces, legacy source, model weights and logs remain local and are not submission artifacts.

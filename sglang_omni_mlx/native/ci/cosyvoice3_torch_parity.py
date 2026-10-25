# SPDX-License-Identifier: Apache-2.0
"""Run controlled native FP16 versus checkpoint-backed Torch component comparisons."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2Model


def require_validation_assertions() -> None:
    """Reject execution modes that remove numerical and weight validation."""
    if not __debug__:
        raise RuntimeError(
            "Validation requires assertions; run without -O/-OO or PYTHONOPTIMIZE"
        )
    else:
        pass


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(native: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    require_validation_assertions()
    assert native.shape == reference.shape
    assert np.isfinite(native).all() and np.isfinite(reference).all()
    actual = native.astype(np.float64).reshape(-1)
    expected = reference.astype(np.float64).reshape(-1)
    difference = actual - expected
    return {
        "max_abs": float(np.abs(difference).max()),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "cosine": float(
            np.dot(actual, expected)
            / (np.linalg.norm(actual) * np.linalg.norm(expected))
        ),
        "violations": int(np.sum(np.abs(difference) > 0.02 + 0.002 * np.abs(expected))),
        "native_top1": int(actual.argmax()),
        "torch_top1": int(expected.argmax()),
        "top1_equal": int(actual.argmax() == expected.argmax()),
        "top5_overlap": len(
            set(np.argsort(actual)[-5:]) & set(np.argsort(expected)[-5:])
        ),
        "top10_overlap": len(
            set(np.argsort(actual)[-10:]) & set(np.argsort(expected)[-10:])
        ),
    }


def verified_weights(
    raw_model: Path, converted_directory: Path
) -> dict[str, torch.Tensor]:
    """Verify every converted tensor against the independently loaded raw checkpoint."""
    require_validation_assertions()
    raw = torch.load(raw_model / "llm.pt", map_location="cpu", weights_only=True)
    converted = load_file(converted_directory / "model.safetensors")
    raw_names = set()
    for name, tensor in sorted(converted.items()):
        raw_name = (
            "llm.model.model." + name.removeprefix("qwen2.")
            if name.startswith("qwen2.")
            else name
        )
        rounded = raw[raw_name].half()
        assert tensor.dtype == torch.float16 and tensor.shape == rounded.shape
        assert torch.equal(tensor, rounded), name
        raw_names.add(raw_name)
    assert set(raw) - raw_names == {"llm.model.lm_head.weight"}
    raw.clear()
    gc.collect()
    return converted


def main() -> int:
    require_validation_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "mps", "all"], required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.output / "report.json").write_text(json.dumps({"status": "incomplete"}))
    if arguments.backend != "cpu":
        if (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or not torch.backends.mps.is_available()
        ):
            parser.error(
                "MPS requires an available device and PYTORCH_ENABLE_MPS_FALLBACK=0; no CPU fallback"
            )
        else:
            pass
    else:
        pass
    torch.set_num_threads(4)
    torch.manual_seed(0)
    converted = verified_weights(arguments.raw_model, arguments.converted)
    config_path = arguments.raw_model / "CosyVoice-BlankEN/config.json"
    base = [151643, 9707, 11, 1879, 0]
    cases = [
        {"name": "text", "text": base, "speech": [], "decode": [42, 1024]},
        {
            "name": "text_speech",
            "text": base,
            "speech": [0, 17, 1234, 6560],
            "decode": [42, 1024],
        },
    ]
    for text_length, speech_length in [(32, 0), (32, 16), (128, 0), (128, 64)]:
        cases.append(
            {
                "name": f"text{text_length}_speech{speech_length}",
                "text": [base[i % len(base)] for i in range(text_length)],
                "speech": [(i * 97 + 17) % 6561 for i in range(speech_length)],
                "decode": [42, 1024, 17, 6560],
            }
        )
    report = {
        "status": "incomplete",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "numpy", "mlx", "mlx-lm"]
        },
        "environment": {
            name: os.environ.get(name)
            for name in [
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "MLX_ENABLE_TF32",
            ]
        },
        "backend": arguments.backend,
        "config": json.loads(config_path.read_text()),
        "converted_config": json.loads(
            (arguments.converted / "config.json").read_text()
        ),
        "files": {
            str(path): checksum(path)
            for path in [
                arguments.raw_model / "llm.pt",
                config_path,
                arguments.converted / "model.safetensors",
                arguments.converted / "config.json",
                arguments.probe,
                Path(__file__),
            ]
        },
        "weights_verified_exact": len(converted),
        "cases": cases,
        "modes": {},
        "rows": [],
        "tolerance": {
            "atol": 0.02,
            "rtol": 0.002,
            "reference": "torch",
            "meaning": "diagnostic only, not universal acceptance criteria",
        },
    }
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2))
    arrays = {}
    for case in cases:
        name = case["name"]
        for step in range(len(case["decode"]) + 1):
            label = (
                f"{name}.native.cached" if step == 0 else f"{name}.native.full{step}"
            )
            logits_path = arguments.output / f"{label}.npy"
            embedding_path = arguments.output / f"{label}.embeddings.npy"
            command = [
                str(arguments.probe.resolve()),
                "--model-path",
                str(arguments.converted),
                "--text-ids",
                json.dumps(case["text"]),
                "--prompt-speech-ids",
                json.dumps(case["speech"] + case["decode"][:step]),
                "--decode-ids",
                json.dumps(case["decode"] if step == 0 else []),
                "--logits-output",
                str(logits_path.resolve()),
                "--embeddings-output",
                str(embedding_path.resolve()),
            ]
            logits_path.unlink(missing_ok=True)
            embedding_path.unlink(missing_ok=True)
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=120
            )
            (arguments.output / f"{label}.process.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    indent=2,
                )
            )
            result.check_returncode()
            summary = json.loads(result.stdout)
            expected_tokens = (
                2
                + len(case["text"])
                + len(case["speech"])
                + (len(case["decode"]) if step == 0 else step)
            )
            assert summary["cache_token_count"] == expected_tokens
            logits = np.load(logits_path)
            if step == 0:
                assert logits.shape == (len(case["decode"]) + 1, 6761)
                arrays[f"{name}.native.cached"] = logits
                arrays[f"{name}.native.embeddings"] = np.load(embedding_path)
                full = [logits[0]]
            else:
                full.append(logits[0])
        arrays[f"{name}.native.full"] = np.stack(full)
        print(f"Native completed: {name}", flush=True)
    backbone_weights = {
        name.removeprefix("qwen2."): tensor
        for name, tensor in converted.items()
        if name.startswith("qwen2.")
    }
    for mode, device, dtype, attention in [
        ("cpu32_eager", "cpu", torch.float32, "eager"),
        ("mps16_eager", "mps", torch.float16, "eager"),
        ("mps32_eager", "mps", torch.float32, "eager"),
        ("mps16_sdpa", "mps", torch.float16, "sdpa"),
    ]:
        if arguments.backend != "all" and device != arguments.backend:
            continue
        else:
            pass
        config = Qwen2Config.from_json_file(str(config_path))
        config._attn_implementation = attention
        model = Qwen2Model(config)
        model.load_state_dict(
            {
                name: tensor.to(dtype=dtype, device=device)
                for name, tensor in backbone_weights.items()
            },
            strict=True,
            assign=True,
        )
        model.to(device=device)
        model.eval()
        speech = converted["speech_embedding.weight"].to(dtype=dtype, device=device)
        head = converted["llm_decoder.weight"].to(dtype=dtype, device=device)
        assert all(
            parameter.device.type == device and parameter.dtype == dtype
            for parameter in model.parameters()
        )
        for name, parameter in model.named_parameters():
            assert torch.equal(
                parameter.detach().cpu().half(), backbone_weights[name]
            ), name
        assert torch.equal(speech.cpu().half(), converted["speech_embedding.weight"])
        assert torch.equal(head.cpu().half(), converted["llm_decoder.weight"])
        report["modes"][mode] = {
            "device": device,
            "dtype": str(dtype),
            "attention": model.config._attn_implementation,
            "class": str(type(model.layers[0].self_attn)),
            "loaded_weights_exact": True,
        }
        with torch.inference_mode():
            for case in cases:
                name = case["name"]
                embeddings = torch.cat(
                    [
                        speech[6561][None, None],
                        model.embed_tokens(torch.tensor([case["text"]], device=device)),
                        speech[6563][None, None],
                        speech[
                            torch.tensor(
                                [case["speech"]], device=device, dtype=torch.long
                            )
                        ],
                    ],
                    dim=1,
                )
                embedding_array = embeddings.float().cpu().numpy()
                assert np.array_equal(
                    embedding_array, arrays[f"{name}.native.embeddings"]
                )
                arrays[f"{name}.{mode}.embeddings"] = embedding_array
                for execution in ["cached", "full"]:
                    cache = None
                    prefix = embeddings
                    logits_rows = []
                    hidden_rows = []
                    for step in range(len(case["decode"]) + 1):
                        if step:
                            next_embedding = speech[case["decode"][step - 1]][
                                None, None
                            ]
                            prefix = torch.cat([prefix, next_embedding], dim=1)
                            current = (
                                next_embedding if execution == "cached" else prefix
                            )
                        else:
                            current = prefix
                        result = model(
                            inputs_embeds=current,
                            attention_mask=torch.ones(
                                (1, prefix.shape[1]), dtype=torch.long, device=device
                            ),
                            past_key_values=cache,
                            use_cache=execution == "cached",
                            return_dict=True,
                        )
                        hidden = result.last_hidden_state[:, -1:, :]
                        logits = torch.nn.functional.linear(hidden, head)
                        assert (
                            hidden.device.type == device
                            and logits.device.type == device
                        )
                        if execution == "cached":
                            cache = result.past_key_values
                            assert cache.get_seq_length() == prefix.shape[1]
                            for layer in cache.layers:
                                assert (
                                    layer.keys.device.type == device
                                    and layer.values.device.type == device
                                )
                        else:
                            pass
                        logits_rows.append(logits.float().cpu().numpy().reshape(6761))
                        hidden_rows.append(hidden.float().cpu().numpy().reshape(896))
                        report["rows"].append(
                            {
                                "case": name,
                                "mode": mode,
                                "execution": execution,
                                "step": step,
                                **metrics(
                                    arrays[f"{name}.native.{execution}"][step],
                                    logits_rows[-1],
                                ),
                            }
                        )
                    arrays[f"{name}.{mode}.{execution}"] = np.stack(logits_rows)
                    arrays[f"{name}.{mode}.{execution}.hidden"] = np.stack(hidden_rows)
                print(f"Torch completed: {mode} {name}", flush=True)
        del (
            model,
            speech,
            head,
            cache,
            result,
            embeddings,
            prefix,
            current,
            hidden,
            logits,
        )
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        else:
            pass
        np.savez(arguments.output / "arrays.npz", **arrays)
        (arguments.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    report["status"] = "completed"
    report["numerical_pass"] = all(row["violations"] == 0 for row in report["rows"])
    report["top1_agreement"] = all(row["top1_equal"] for row in report["rows"])
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Completed; numerical_pass={report['numerical_pass']}, top1_agreement={report['top1_agreement']}"
    )
    return 0 if report["numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass

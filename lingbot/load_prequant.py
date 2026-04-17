#!/usr/bin/env python3
"""
Load pre-quantized bitsandbytes NF4 models WITHOUT loading the full dense model
or entire safetensors file into host RAM.

This version fixes:
- host RAM OOM from full load_file()
- meta-model no-op state_dict loads
- non-linear params/biases staying meta / wrong dtype
- conv3d bf16-vs-float bias mismatch during inference
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState

import sys
sys.path.insert(0, str(Path(__file__).parent))

from wan.modules.model import WanModel


# ============================================================
# Meta construction utilities
# ============================================================

class _DefaultDeviceCtx:
    """
    Context manager for torch.set_default_device if available.
    Used to build large modules on 'meta' without allocating storage.
    """
    def __init__(self, device: str):
        self.device = device
        self._has = hasattr(torch, "set_default_device")
        self._prev = None

    def __enter__(self):
        if self._has:
            self._prev = getattr(torch, "get_default_device", lambda: None)()
            torch.set_default_device(self.device)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._has and self._prev is not None:
            torch.set_default_device(self._prev)


def build_model_from_config_meta(config: Dict[str, Any]) -> WanModel:
    """
    Build WanModel on META device so dense params don't allocate CPU RAM.
    """
    with torch.no_grad():
        with _DefaultDeviceCtx("meta"):
            model = WanModel(
                model_type=config.get("model_type", "i2v"),
                patch_size=tuple(config.get("patch_size", (1, 2, 2))),
                text_len=config.get("text_len", 512),
                in_dim=config.get("in_dim", 16),
                dim=config.get("dim", 2048),
                ffn_dim=config.get("ffn_dim", 8192),
                freq_dim=config.get("freq_dim", 256),
                text_dim=config.get("text_dim", 4096),
                out_dim=config.get("out_dim", 16),
                num_heads=config.get("num_heads", 16),
                num_layers=config.get("num_layers", 32),
                window_size=tuple(config.get("window_size", (-1, -1))),
                qk_norm=config.get("qk_norm", True),
                cross_attn_norm=config.get("cross_attn_norm", True),
                eps=config.get("eps", 1e-6),
            )
    return model


# ============================================================
# Model surgery
# ============================================================

def replace_linears_with_bnb_nf4(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    compress_statistics: bool = True,
    quant_type: str = "nf4",
) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """
    Replace nn.Linear layers with bnb.nn.Linear4bit layers.
    """
    replaced = 0
    layer_shapes: Dict[str, Tuple[int, int]] = {}

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for name, module in linear_layers:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        layer_shapes[name] = (module.in_features, module.out_features)

        nf4_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )

        setattr(parent, child_name, nf4_linear)
        replaced += 1

    return replaced, layer_shapes


# ============================================================
# Quantized weight reconstruction
# ============================================================

def reconstruct_params4bit_from_components(
    weight_components: Dict[str, torch.Tensor],
    device: str,
) -> bnb.nn.Params4bit:
    """
    Reconstruct bnb Params4bit from serialized quantized components.
    """
    qs_dict = {
        "absmax": weight_components["absmax"],
        "quant_map": weight_components["quant_map"],
    }

    if "nested_absmax" in weight_components:
        qs_dict["nested_absmax"] = weight_components["nested_absmax"]
        qs_dict["nested_quant_map"] = weight_components["nested_quant_map"]

    if "quant_state_data" in weight_components:
        qs_dict["quant_state.bitsandbytes__nf4"] = weight_components["quant_state_data"]

    quant_state = QuantState.from_dict(qs_dict, device=torch.device(device))
    quantized_weight = weight_components["weight"].to(device)

    return bnb.nn.Params4bit(
        data=quantized_weight,
        requires_grad=False,
        quant_state=quant_state,
        bnb_quantized=True,
    )


def _is_quant_linear_weight_key(all_keys_set: set, key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    base = key[:-7]
    required = [
        f"{base}.weight.absmax",
        f"{base}.weight.quant_map",
        f"{base}.weight.quant_state.bitsandbytes__nf4",
    ]
    return all(k in all_keys_set for k in required)


def _cast_tensor_for_model(t: torch.Tensor, compute_dtype: torch.dtype, device: str) -> torch.Tensor:
    """
    Cast floating tensors to compute_dtype and move to target device.
    Non-floating tensors keep their dtype.
    """
    if t.is_floating_point():
        return t.to(device=device, dtype=compute_dtype)
    return t.to(device=device)


def _assert_no_meta_tensors(model: nn.Module) -> None:
    meta_params = [name for name, p in model.named_parameters() if getattr(p, "is_meta", False)]
    meta_buffers = [name for name, b in model.named_buffers() if getattr(b, "is_meta", False)]

    if meta_params or meta_buffers:
        preview = (meta_params + meta_buffers)[:20]
        raise RuntimeError(
            "Model still contains meta tensors after loading. "
            f"Examples: {preview}"
        )


# ============================================================
# Main loading path
# ============================================================

def load_quantized_model(
    model_dir: str,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> WanModel:
    """
    Load a pre-quantized WanModel from a directory.

    Improvements:
    - build on meta
    - stream safetensors
    - use assign=True for meta params
    - cast non-linear floating tensors to compute_dtype
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    meta_path = model_dir / "quantization_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        quant_config = meta.get("quant", {})
        compute_dtype_str = quant_config.get("compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

    safetensors_path = model_dir / "model.safetensors"
    pt_path = model_dir / "model.pt"

    if safetensors_path.exists():
        weights_path = safetensors_path
        is_safe = True
    elif pt_path.exists():
        weights_path = pt_path
        is_safe = False
    else:
        raise FileNotFoundError(
            f"No weights found in {model_dir}. Expected model.safetensors or model.pt"
        )

    print(f"Loading pre-quantized model from {model_dir}")
    print(f"  Config: {config_path}")
    print(f"  Weights: {weights_path}")

    # 1) Build the large model on meta so dense params do not allocate real RAM
    model = build_model_from_config_meta(config)

    # 2) Replace Linear -> Linear4bit
    replaced, layer_shapes = replace_linears_with_bnb_nf4(
        model,
        compute_dtype=compute_dtype,
    )
    print(f"  Replaced {replaced} linear layers with bnb.Linear4bit")

    # 3) Load weights
    if is_safe:
        _load_quantized_state_safetensors_stream(
            model=model,
            weights_path=str(weights_path),
            layer_shapes=layer_shapes,
            device=device,
            compute_dtype=compute_dtype,
        )
    else:
        _load_quantized_state_pt(
            model=model,
            weights_path=str(weights_path),
            layer_shapes=layer_shapes,
            device=device,
            compute_dtype=compute_dtype,
        )

    model.eval()
    model.requires_grad_(False)

    _assert_no_meta_tensors(model)

    print(f"  Model ready on {device}")
    return model


# ============================================================
# Safetensors streamed loader
# ============================================================

def _load_quantized_state_safetensors_stream(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str,
    compute_dtype: torch.dtype,
) -> None:
    """
    Stream safetensors and assign weights without loading the full file into RAM.
    """
    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

    all_keys_set = set(all_keys)

    quant_bases = set()
    non_linear_keys = []

    for k in all_keys:
        if k.endswith(".weight") and _is_quant_linear_weight_key(all_keys_set, k):
            quant_bases.add(k[:-7])
        elif (
            k.endswith(".weight.absmax")
            or k.endswith(".weight.quant_map")
            or k.endswith(".weight.quant_state.bitsandbytes__nf4")
            or k.endswith(".weight.nested_absmax")
            or k.endswith(".weight.nested_quant_map")
        ):
            continue
        else:
            non_linear_keys.append(k)

    # 1) Load quantized linear weights layer-by-layer
    loaded_count = 0
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for base in sorted(quant_bases):
            mod = model.get_submodule(base)
            if not isinstance(mod, bnb.nn.Linear4bit):
                continue

            comps: Dict[str, torch.Tensor] = {
                "weight": f.get_tensor(f"{base}.weight"),
                "absmax": f.get_tensor(f"{base}.weight.absmax"),
                "quant_map": f.get_tensor(f"{base}.weight.quant_map"),
                "quant_state_data": f.get_tensor(f"{base}.weight.quant_state.bitsandbytes__nf4"),
            }

            nk1 = f"{base}.weight.nested_absmax"
            nk2 = f"{base}.weight.nested_quant_map"
            if nk1 in all_keys_set and nk2 in all_keys_set:
                comps["nested_absmax"] = f.get_tensor(nk1)
                comps["nested_quant_map"] = f.get_tensor(nk2)

            mod.weight = reconstruct_params4bit_from_components(comps, device=device)

            bias_key = f"{base}.bias"
            if bias_key in all_keys_set and mod.bias is not None:
                bias_tensor = _cast_tensor_for_model(f.get_tensor(bias_key), compute_dtype, device)
                mod.bias = nn.Parameter(bias_tensor, requires_grad=False)

            loaded_count += 1

            for t in comps.values():
                del t

    # 2) Load non-linear params/buffers in batches with assign=True
    # This is the crucial fix for meta no-op warnings.
    batch = {}
    batch_max = 256

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for k in non_linear_keys:
            batch[k] = _cast_tensor_for_model(f.get_tensor(k), compute_dtype, device)

            if len(batch) >= batch_max:
                model.load_state_dict(batch, strict=False, assign=True)
                batch.clear()

        if batch:
            model.load_state_dict(batch, strict=False, assign=True)
            batch.clear()

    print(f"  Loaded {loaded_count} quantized linear layers (streamed safetensors)")


# ============================================================
# .pt fallback loader
# ============================================================

def _load_quantized_state_pt(
    model: nn.Module,
    weights_path: str,
    layer_shapes: Dict[str, Tuple[int, int]],
    device: str,
    compute_dtype: torch.dtype,
) -> None:
    """
    Fallback for .pt checkpoints.
    This can still be RAM-heavier than safetensors streaming, but supports assign=True.
    """
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)

    weight_components = defaultdict(dict)
    other_keys: Dict[str, torch.Tensor] = {}

    quant_suffixes = [
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__nf4",
    ]

    for key, tensor in sd.items():
        base_key = None
        component = None

        if ".weight.absmax" in key:
            base_key = key.replace(".weight.absmax", "")
            component = "absmax"
        elif ".weight.quant_map" in key:
            base_key = key.replace(".weight.quant_map", "")
            component = "quant_map"
        elif ".weight.nested_absmax" in key:
            base_key = key.replace(".weight.nested_absmax", "")
            component = "nested_absmax"
        elif ".weight.nested_quant_map" in key:
            base_key = key.replace(".weight.nested_quant_map", "")
            component = "nested_quant_map"
        elif ".weight.quant_state.bitsandbytes__nf4" in key:
            base_key = key.replace(".weight.quant_state.bitsandbytes__nf4", "")
            component = "quant_state_data"
        elif key.endswith(".weight"):
            potential_base = key[:-7]
            has_quant_metadata = any(
                f"{potential_base}.weight{suffix}" in sd for suffix in quant_suffixes
            )
            if has_quant_metadata:
                base_key = potential_base
                component = "weight"
            else:
                other_keys[key] = tensor
                continue
        else:
            other_keys[key] = tensor
            continue

        if base_key and component:
            weight_components[base_key][component] = tensor

    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) and name in weight_components:
            comps = weight_components[name]
            if "weight" in comps:
                module.weight = reconstruct_params4bit_from_components(comps, device=device)
                loaded_count += 1

            bias_key = f"{name}.bias"
            if bias_key in other_keys and module.bias is not None:
                bias_tensor = _cast_tensor_for_model(other_keys[bias_key], compute_dtype, device)
                module.bias = nn.Parameter(bias_tensor, requires_grad=False)

    if other_keys:
        casted = {
            k: _cast_tensor_for_model(v, compute_dtype, device)
            for k, v in other_keys.items()
        }
        model.load_state_dict(casted, strict=False, assign=True)

    print(f"  Loaded {loaded_count} quantized linear layers (.pt)")


# ============================================================
# Verification
# ============================================================

def verify_quantized_model(model: nn.Module) -> Dict[str, Any]:
    total_params = 0
    quantized_params = 0
    linear4bit_count = 0
    regular_linear_count = 0

    for _, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear4bit_count += 1
            if hasattr(module.weight, "quant_state") and module.weight.quant_state is not None:
                quantized_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            regular_linear_count += 1

    for p in model.parameters():
        total_params += p.numel()

    return {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "linear4bit_count": linear4bit_count,
        "regular_linear_count": regular_linear_count,
        "is_quantized": linear4bit_count > 0 and regular_linear_count == 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test loading pre-quantized model")
    parser.add_argument("model_dir", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = load_quantized_model(args.model_dir, device=args.device)
    info = verify_quantized_model(model)

    print("\nVerification:")
    print(f"  Linear4bit layers: {info['linear4bit_count']}")
    print(f"  Regular Linear layers: {info['regular_linear_count']}")
    print(f"  Is properly quantized: {info['is_quantized']}")
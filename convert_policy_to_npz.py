from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _extract_sort_index(name: str) -> Tuple[int, str]:
    nums = re.findall(r"\d+", name)
    idx = int(nums[-1]) if nums else 10**9
    return idx, name


def _pack_npz(
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    out_path: Path,
    act_scale: float,
    act_bias: float,
    tanh_out: bool,
) -> None:
    if not weights or not biases or len(weights) != len(biases):
        raise ValueError("无效网络参数：weights/biases 为空或层数不一致。")

    action_dim = biases[-1].shape[0]
    payload: Dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(zip(weights, biases)):
        payload[f"W{i}"] = w.astype(np.float32)
        payload[f"b{i}"] = b.astype(np.float32)

    payload["act_scale"] = np.full(action_dim, act_scale, dtype=np.float32)
    payload["act_bias"] = np.full(action_dim, act_bias, dtype=np.float32)
    payload["tanh_out"] = np.array(1 if tanh_out else 0, dtype=np.int32)

    np.savez(out_path, **payload)


def _normalize_torch_obj(obj: object) -> Dict[str, object]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj

    # 可能是 nn.Module 或 torch.jit.ScriptModule
    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
        if isinstance(sd, dict):
            return sd

    raise ValueError("无法从该 PyTorch 文件解析 state_dict。")


def _load_torch_layers(torch_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 PyTorch，请先安装 torch。") from exc

    obj = torch.load(torch_path, map_location="cpu")
    state_dict = _normalize_torch_obj(obj)

    weight_items = []
    bias_items = []
    for k, v in state_dict.items():
        if not hasattr(v, "detach"):
            continue
        if k.endswith("weight") and getattr(v, "ndim", 0) == 2:
            weight_items.append((k, v.detach().cpu().numpy()))
        elif k.endswith("bias") and getattr(v, "ndim", 0) == 1:
            bias_items.append((k, v.detach().cpu().numpy()))

    if not weight_items or not bias_items:
        raise ValueError("未在 state_dict 中找到二维 weight 和一维 bias。")

    weight_items.sort(key=lambda x: _extract_sort_index(x[0]))
    bias_items.sort(key=lambda x: _extract_sort_index(x[0]))

    if len(weight_items) != len(bias_items):
        raise ValueError("weight/bias 数量不一致，请检查模型结构。")

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for (_, w), (_, b) in zip(weight_items, bias_items):
        # PyTorch Linear: [out, in]；stm2sim.py 期望 [in, out]
        wt = np.asarray(w, dtype=np.float32).T
        bt = np.asarray(b, dtype=np.float32)
        if wt.shape[1] != bt.shape[0]:
            raise ValueError(
                f"层参数维度不匹配: W={wt.shape}, b={bt.shape}。"
            )
        weights.append(wt)
        biases.append(bt)

    return weights, biases


def _iter_onnx_initializers(onnx_model: object) -> Iterable[Tuple[str, np.ndarray]]:
    try:
        onnx = importlib.import_module("onnx")
        numpy_helper = onnx.numpy_helper
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 onnx，请先安装 onnx。") from exc

    for init in onnx_model.graph.initializer:
        yield init.name, numpy_helper.to_array(init)


def _load_onnx_layers(onnx_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    try:
        onnx = importlib.import_module("onnx")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 onnx，请先安装 onnx。") from exc

    model = onnx.load(str(onnx_path))
    init_map = {k: np.asarray(v) for k, v in _iter_onnx_initializers(model)}

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []

    for node in model.graph.node:
        if node.op_type not in {"Gemm", "MatMul"}:
            continue

        # 常见导出：Gemm(x, W, b)
        w = None
        b = None
        for name in node.input:
            if name in init_map:
                arr = init_map[name]
                if arr.ndim == 2 and w is None:
                    w = arr
                elif arr.ndim == 1 and b is None:
                    b = arr

        if w is None:
            continue

        # 转成 [in, out]
        if w.shape[0] < w.shape[1]:
            # 常见 [out, in]
            w = w.T
        if b is None:
            b = np.zeros(w.shape[1], dtype=np.float32)

        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if w.shape[1] != b.shape[0]:
            raise ValueError(f"ONNX 层参数维度不匹配: W={w.shape}, b={b.shape}")
        weights.append(w)
        biases.append(b)

    if not weights:
        raise ValueError("未在 ONNX 图中解析到线性层（Gemm/MatMul）。")

    return weights, biases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 PyTorch/ONNX 策略转换为 stm2sim.py 可用 .npz")
    parser.add_argument("--input", required=True, help="输入策略文件路径 (.pt/.pth/.onnx)")
    parser.add_argument("--output", default="offline_policy.npz", help="输出 .npz 路径")
    parser.add_argument("--act-scale", type=float, default=1.0, help="输出动作缩放")
    parser.add_argument("--act-bias", type=float, default=0.0, help="输出动作偏置")
    parser.add_argument("--no-tanh-out", action="store_true", help="禁用输出层 tanh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    suffix = in_path.suffix.lower()

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    if suffix in {".pt", ".pth", ".ckpt"}:
        weights, biases = _load_torch_layers(in_path)
    elif suffix == ".onnx":
        weights, biases = _load_onnx_layers(in_path)
    else:
        raise ValueError("仅支持 .pt/.pth/.ckpt/.onnx")

    _pack_npz(
        weights=weights,
        biases=biases,
        out_path=out_path,
        act_scale=args.act_scale,
        act_bias=args.act_bias,
        tanh_out=not args.no_tanh_out,
    )

    print(f"转换完成: {in_path} -> {out_path}")
    print(f"网络层数: {len(weights)}")
    print(f"观测输入维度: {weights[0].shape[0]}")
    print(f"动作输出维度: {biases[-1].shape[0]}")


if __name__ == "__main__":
    main()

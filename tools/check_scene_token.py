#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Check scene token distribution for offline BEV export")
    parser.add_argument(
        "--bev-dir",
        type=str,
        default="",
        help="Path to offline BEV directory. If empty, infer from --split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help="Optional split name to infer default bev dir.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default="bev_feature.json",
        help="Metadata json filename under bev dir.",
    )
    parser.add_argument(
        "--strict-filename",
        action="store_true",
        help="Require filename suffix to be _<frame_nbr>_<scene_token>.pth",
    )
    parser.add_argument(
        "--check-shape",
        action="store_true",
        help="Load each pth and verify strict single-frame shape [1,HW,C].",
    )
    parser.add_argument("--expected-t", type=int, default=1)
    parser.add_argument("--expected-hw", type=int, default=40000)
    parser.add_argument("--expected-c", type=int, default=256)
    return parser.parse_args()


def infer_dir(split):
    mapping = {
        "train": "data/nuscenes/bev_offline_features_train",
        "val": "data/nuscenes/bev_offline_features_val",
        "test": "data/nuscenes/bev_offline_features_test",
        "": "data/nuscenes/bev_offline_features",
    }
    return Path(mapping[split])


def load_tensor(path):
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        if "bev_feature" in loaded:
            loaded = loaded["bev_feature"]
        elif "state_dict" in loaded and isinstance(loaded["state_dict"], torch.Tensor):
            loaded = loaded["state_dict"]
        else:
            raise ValueError("unsupported payload")
    if not isinstance(loaded, torch.Tensor):
        raise ValueError("payload is not tensor")
    if loaded.dim() != 3:
        raise ValueError(f"expected 3D tensor [1,HW,C], got {tuple(loaded.shape)}")
    if loaded.shape[0] != 1:
        raise ValueError(f"legacy multi-frame tensor is not supported: {tuple(loaded.shape)}")
    return loaded

args = parse_args()

bev_dir = Path(args.bev_dir) if args.bev_dir else infer_dir(args.split)
meta = bev_dir / args.metadata_name

if not meta.exists():
    raise SystemExit(f"[FAIL] metadata not found: {meta}")

items = json.loads(meta.read_text())
if not isinstance(items, list):
    raise SystemExit("[FAIL] bev_feature.json is not a list")

cnt = Counter()
missing_scene = 0
missing_files = []
bad_names = []
bad_shapes = []
for it in items:
    fn = it.get("filename", "")
    if not fn:
        missing_files.append("<missing filename field>")
        continue

    p = bev_dir / fn
    if not p.exists():
        missing_files.append(fn)

    if args.strict_filename:
        scene_token = it.get("scene_token", "")
        frame_nbr = it.get("frame_nbr", "")
        suffix = f"_{frame_nbr}_{scene_token}.pth"
        if not (scene_token and str(fn).endswith(suffix)):
            bad_names.append(fn)

    if args.check_shape and p.exists():
        try:
            tensor = load_tensor(p)
            shape = tuple(tensor.shape)
            expected = (args.expected_t, args.expected_hw, args.expected_c)
            if shape != expected:
                bad_shapes.append((fn, shape))
        except Exception as exc:  # noqa: BLE001
            bad_shapes.append((fn, str(exc)))

    s = it.get("scene_token")
    if not s:
        missing_scene += 1
        continue
    cnt[s] += 1

print(f"[INFO] total entries: {len(items)}")
print(f"[INFO] unique scenes: {len(cnt)}")
print(f"[INFO] entries missing scene_token: {missing_scene}")
print(f"[INFO] missing files: {len(missing_files)}")
if args.strict_filename:
    print(f"[INFO] bad filename pattern: {len(bad_names)}")
if args.check_shape:
    print(f"[INFO] bad tensor shapes: {len(bad_shapes)}")

if cnt:
    vals = list(cnt.values())
    print(f"[INFO] per-scene min/max/avg: {min(vals)}/{max(vals)}/{sum(vals)/len(vals):.2f}")
    print("[INFO] top 10 scenes by count:")
    for scene, n in cnt.most_common(10):
        print(f" - {scene}: {n}")

if missing_files:
    print("[DETAIL] first 20 missing files:")
    for x in missing_files[:20]:
        print(" -", x)

if bad_names:
    print("[DETAIL] first 20 bad filename patterns:")
    for x in bad_names[:20]:
        print(" -", x)

if bad_shapes:
    print("[DETAIL] first 20 bad tensor shapes:")
    for fn, shape in bad_shapes[:20]:
        print(f" - {fn}: {shape}")

# 可选：导出完整统计到文件
out = bev_dir / "scene_stats.json"
out.write_text(json.dumps(
    {
        "total_entries": len(items),
        "unique_scenes": len(cnt),
        "missing_scene_token": missing_scene,
        "missing_files": len(missing_files),
        "bad_filename_pattern": len(bad_names),
        "bad_tensor_shapes": len(bad_shapes),
        "counts": cnt,
    },
    ensure_ascii=False, indent=2, default=lambda x: dict(x)
))
print(f"[OK] wrote scene stats: {out}")

if missing_scene or missing_files or bad_names or bad_shapes:
    raise SystemExit("[FAIL] scene token check failed")

print("[PASS] scene token check passed")
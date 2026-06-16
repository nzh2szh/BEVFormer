#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Check offline BEV export integrity")
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
        "--check-shape",
        action="store_true",
        help="Load each pth and validate shape.",
    )
    parser.add_argument("--expected-t", type=int, default=1)
    parser.add_argument("--expected-hw", type=int, default=40000)
    parser.add_argument("--expected-c", type=int, default=256)
    parser.add_argument(
        "--strict-filename",
        action="store_true",
        help="Require filename suffix to be _<frame_nbr>_<scene_token>.pth",
    )
    parser.add_argument(
        "--scene-json",
        type=str,
        default="",
        help="Optional scene.json path. If set, validates frame_token/frame_nbr mapping via sample.json chain.",
    )
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
    # Strict mode: only single-frame [1, HW, C] is accepted.
    if loaded.dim() != 3:
        raise ValueError(f"expected 3D tensor [1,HW,C], got {tuple(loaded.shape)}")
    if loaded.shape[0] != 1:
        raise ValueError(f"legacy multi-frame tensor is not supported: {tuple(loaded.shape)}")
    return loaded


def build_scene_sample_order(scene_json_path):
    scene_path = Path(scene_json_path)
    sample_path = scene_path.parent / "sample.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"scene.json not found: {scene_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"sample.json not found next to scene.json: {sample_path}")

    scenes = json.loads(scene_path.read_text())
    samples = json.loads(sample_path.read_text())
    sample_by_token = {}
    for rec in samples:
        if isinstance(rec, dict):
            tok = rec.get("token", "")
            if tok:
                sample_by_token[tok] = rec

    order = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_token = scene.get("token", "")
        first_token = scene.get("first_sample_token", "")
        last_token = scene.get("last_sample_token", "")
        if not scene_token or not first_token:
            continue

        idx = 0
        tok = first_token
        visited = set()
        while tok:
            if tok in visited:
                break
            visited.add(tok)
            order[f"{scene_token}::{tok}"] = idx
            if tok == last_token:
                break
            rec = sample_by_token.get(tok)
            if rec is None:
                break
            tok = rec.get("next", "")
            idx += 1
    return order


args = parse_args()

bev_dir = Path(args.bev_dir) if args.bev_dir else infer_dir(args.split)
meta = bev_dir / args.metadata_name

if not meta.exists():
    raise SystemExit(f"[FAIL] metadata not found: {meta}")

items = json.loads(meta.read_text())
if not isinstance(items, list):
    raise SystemExit("[FAIL] bev_feature.json is not a list")

missing = []
seen = set()
dup = 0
bad_names = []
bad_shapes = []
bad_fields = []
dup_scene_frame = 0
dup_scene_token = 0
scene_frame_seen = set()
scene_token_seen = set()

scene_sample_order = None
bad_scene_chain = []
if args.scene_json:
    scene_sample_order = build_scene_sample_order(args.scene_json)

for i, it in enumerate(items):
    scene_token = it.get("scene_token", "")
    frame_nbr = it.get("frame_nbr", None)
    frame_token = it.get("frame_token", "")
    if not scene_token or frame_nbr is None or frame_token == "":
        bad_fields.append(f"(index={i}) scene_token/frame_nbr/frame_token missing")

    fn = it.get("filename")
    if not fn:
        missing.append(f"(index={i}) missing filename field")
        continue

    if args.strict_filename:
        suffix = f"_{frame_nbr}_{scene_token}.pth"
        if not (scene_token and str(fn).endswith(suffix)):
            bad_names.append(fn)

    if scene_token and frame_nbr is not None:
        try:
            sf_key = (scene_token, int(frame_nbr))
            if sf_key in scene_frame_seen:
                dup_scene_frame += 1
            scene_frame_seen.add(sf_key)
        except Exception:  # noqa: BLE001
            bad_fields.append(f"(index={i}) frame_nbr is not int-compatible: {frame_nbr}")

    if scene_token and frame_token:
        st_key = (scene_token, frame_token)
        if st_key in scene_token_seen:
            dup_scene_token += 1
        scene_token_seen.add(st_key)

    if scene_sample_order is not None and scene_token and frame_token and frame_nbr is not None:
        key = f"{scene_token}::{frame_token}"
        expected = scene_sample_order.get(key, None)
        if expected is None:
            bad_scene_chain.append(f"{fn}: frame_token not found in scene/sample chain")
        else:
            try:
                if int(frame_nbr) != int(expected):
                    bad_scene_chain.append(
                        f"{fn}: frame_nbr={frame_nbr}, expected={expected} by scene/sample chain"
                    )
            except Exception:  # noqa: BLE001
                bad_scene_chain.append(f"{fn}: frame_nbr is not int-compatible: {frame_nbr}")

    p = bev_dir / fn
    if not p.exists():
        missing.append(fn)
    elif args.check_shape:
        try:
            tensor = load_tensor(p)
            shape = tuple(tensor.shape)
            expected = (args.expected_t, args.expected_hw, args.expected_c)
            if shape != expected:
                bad_shapes.append((fn, shape))
        except Exception as exc:  # noqa: BLE001
            bad_shapes.append((fn, str(exc)))

    if fn in seen:
        dup += 1
    seen.add(fn)

print(f"[OK] metadata: {meta}")
print(f"[INFO] entries: {len(items)}")
print(f"[INFO] missing files: {len(missing)}")
print(f"[INFO] duplicate filenames: {dup}")
print(f"[INFO] bad filename pattern: {len(bad_names)}")
print(f"[INFO] bad required fields: {len(bad_fields)}")
print(f"[INFO] duplicate (scene_token, frame_nbr): {dup_scene_frame}")
print(f"[INFO] duplicate (scene_token, frame_token): {dup_scene_token}")
if scene_sample_order is not None:
    print(f"[INFO] scene/sample chain mismatches: {len(bad_scene_chain)}")
if args.check_shape:
    print(f"[INFO] bad tensor shapes: {len(bad_shapes)}")

if missing:
    print("[DETAIL] first 20 missing:")
    for x in missing[:20]:
        print(" -", x)

if bad_names:
    print("[DETAIL] first 20 bad filename patterns:")
    for x in bad_names[:20]:
        print(" -", x)

if bad_fields:
    print("[DETAIL] first 20 bad required fields:")
    for x in bad_fields[:20]:
        print(" -", x)

if bad_scene_chain:
    print("[DETAIL] first 20 scene/sample chain mismatches:")
    for x in bad_scene_chain[:20]:
        print(" -", x)

if bad_shapes:
    print("[DETAIL] first 20 bad tensor shapes:")
    for fn, shape in bad_shapes[:20]:
        print(f" - {fn}: {shape}")

if len(items) == 0 or missing or bad_names or bad_shapes or bad_fields or dup_scene_frame or dup_scene_token or bad_scene_chain:
    raise SystemExit("[FAIL] export integrity check failed")
print("[PASS] export integrity check passed")
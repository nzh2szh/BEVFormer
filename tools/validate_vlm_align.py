import argparse
import importlib
import json
import os

import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    """Parse CLI arguments for retrieval validation.

    Supported checkpoint input styles:
    1) Legacy single checkpoint: positional `checkpoint` only.
    2) Two-stage overlay: `--base-ckpt` then `--align-ckpt`.
    3) Semi-implicit two-stage: `--align-ckpt` plus positional checkpoint
       or config `load_from` as base.
    """
    parser = argparse.ArgumentParser(description='Validate BEVFormer VLM alignment model')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='single checkpoint path (legacy mode, loads once)')
    parser.add_argument(
        '--base-ckpt',
        default=None,
        help='base/full model checkpoint path loaded first')
    parser.add_argument(
        '--align-ckpt',
        default=None,
        help='alignment-only checkpoint path loaded after base checkpoint')
    parser.add_argument(
        '--load-report',
        default=None,
        help='optional path to dump checkpoint load diagnostics as json')
    parser.add_argument(
        '--dump-embedding-diagnostics',
        default=None,
        help='optional path to dump vision/text embedding retrieval diagnostics as json')
    parser.add_argument(
        '--max-align-missing-keys',
        type=int,
        default=None,
        help='fail if align missing keys count is greater than this value')
    parser.add_argument(
        '--fail-on-unexpected-keys',
        action='store_true',
        help='fail if align unexpected keys count is greater than 0')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='val batch size per gpu')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='num workers per gpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in config, key=value format')
    return parser.parse_args()


def _load_alignment_checkpoint(model, checkpoint_path):
    """Overlay alignment weights onto an already initialized base model.

    The alignment checkpoint usually contains only trainable alignment modules,
    so strict loading is intentionally disabled.

    Returns:
        tuple[list[str], list[str]]: (missing_keys, unexpected_keys)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(
        'Loaded align checkpoint: {} (missing_keys={}, unexpected_keys={})'.format(
            checkpoint_path,
            len(missing_keys),
            len(unexpected_keys),
        )
    )
    return missing_keys, unexpected_keys


def _dump_load_report(report, report_path):
    """Persist checkpoint-load diagnostics for reproducibility/auditing."""
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Saved load report: {}'.format(report_path))


def _check_load_report_or_fail(args, report):
    """Apply optional strict checks and fail fast for CI/pipeline use.

    Rules are only meaningful when an alignment checkpoint is overlaid.
    """
    align_ckpt = report.get('align_ckpt', None)
    if align_ckpt is None:
        print('Skip strict load checks: align checkpoint not used in this run.')
        return

    missing_cnt = len(report.get('align_missing_keys', []))
    unexpected_cnt = len(report.get('align_unexpected_keys', []))

    if args.max_align_missing_keys is not None and missing_cnt > args.max_align_missing_keys:
        raise SystemExit(
            'align missing_keys {} > max_align_missing_keys {}. '.format(
                missing_cnt,
                args.max_align_missing_keys,
            )
            + 'Failing as requested.'
        )

    if args.fail_on_unexpected_keys and unexpected_cnt > 0:
        raise SystemExit(
            'align unexpected_keys {} > 0 with --fail-on-unexpected-keys. '.format(
                unexpected_cnt,
            )
            + 'Failing as requested.'
        )


def _tensor_stats(values):
    """Return compact numeric stats for a 1D tensor."""
    if values.numel() == 0:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'p05': None,
            'p25': None,
            'median': None,
            'p75': None,
            'p95': None,
            'p99': None,
            'max': None,
        }
    values = values.detach().cpu().float().view(-1)
    quantiles = torch.quantile(
        values,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=values.dtype),
    )
    return {
        'count': int(values.numel()),
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
        'min': float(values.min().item()),
        'p05': float(quantiles[0].item()),
        'p25': float(quantiles[1].item()),
        'median': float(quantiles[2].item()),
        'p75': float(quantiles[3].item()),
        'p95': float(quantiles[4].item()),
        'p99': float(quantiles[5].item()),
        'max': float(values.max().item()),
    }


def _offdiag_values(matrix):
    """Return off-diagonal values from a square matrix."""
    if matrix.numel() == 0 or matrix.size(0) != matrix.size(1):
        return matrix.new_empty((0,))
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _frequency_report(indices, labels, topk=10):
    """Count how often each label appears in an index tensor."""
    if indices.numel() == 0 or len(labels) == 0:
        return {
            'count': 0,
            'unique': 0,
            'max_frequency': 0,
            'max_ratio': 0.0,
            'top_labels': [],
        }

    counts = {}
    for idx in indices.detach().cpu().view(-1).tolist():
        idx = int(idx)
        if idx < 0 or idx >= len(labels):
            continue
        key = str(labels[idx])
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return {
            'count': 0,
            'unique': 0,
            'max_frequency': 0,
            'max_ratio': 0.0,
            'top_labels': [],
        }

    total = int(sum(counts.values()))
    top_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:topk]
    return {
        'count': total,
        'unique': int(len(counts)),
        'max_frequency': int(top_items[0][1]),
        'max_ratio': float(top_items[0][1] / float(total)),
        'top_labels': [
            {
                'label': key,
                'count': int(val),
                'ratio': float(val / float(total)),
            }
            for key, val in top_items
        ],
    }


def _feature_geometry_report(mat, max_components=8):
    """Summarize feature concentration with centroid and PCA statistics."""
    if mat.numel() == 0 or mat.dim() != 2:
        return {
            'count': 0,
            'dim': None,
            'centroid_norm': None,
            'cosine_to_centroid': _tensor_stats(mat.new_empty((0,))),
            'centered_l2_norm': _tensor_stats(mat.new_empty((0,))),
            'pca': {
                'rank': 0,
                'explained_variance_ratio': [],
                'cumulative_explained_variance_ratio': [],
                'effective_rank': None,
            },
        }

    mat = mat.detach().cpu().float()
    centered = mat - mat.mean(dim=0, keepdim=True)
    centroid = mat.mean(dim=0)
    centroid_norm = float(centroid.norm().item())

    if centroid_norm > 0:
        centroid_dir = centroid / centroid.norm().clamp(min=1e-12)
        cosine_to_centroid = mat @ centroid_dir
    else:
        cosine_to_centroid = mat.new_zeros((mat.size(0),))

    centered_l2 = centered.norm(dim=1)

    if mat.size(0) >= 2 and mat.size(1) >= 1:
        singular_vals = torch.linalg.svdvals(centered)
        variances = singular_vals.square()
        total_var = variances.sum()
        if float(total_var.item()) > 0:
            evr = variances / total_var
            take = min(int(max_components), int(evr.numel()))
            evr_take = evr[:take]
            cum_take = torch.cumsum(evr_take, dim=0)
            probs = evr[evr > 0]
            effective_rank = float(torch.exp(-(probs * probs.log()).sum()).item()) if probs.numel() > 0 else None
        else:
            evr_take = mat.new_zeros((0,))
            cum_take = mat.new_zeros((0,))
            effective_rank = None
    else:
        evr_take = mat.new_zeros((0,))
        cum_take = mat.new_zeros((0,))
        effective_rank = None

    return {
        'count': int(mat.size(0)),
        'dim': int(mat.size(1)),
        'centroid_norm': centroid_norm,
        'cosine_to_centroid': _tensor_stats(cosine_to_centroid),
        'centered_l2_norm': _tensor_stats(centered_l2),
        'pca': {
            'rank': int(min(mat.size(0), mat.size(1))),
            'explained_variance_ratio': [float(x) for x in evr_take.tolist()],
            'cumulative_explained_variance_ratio': [float(x) for x in cum_take.tolist()],
            'effective_rank': effective_rank,
        },
    }


def _dump_embedding_diagnostics(
        dump_path,
        sim,
        positive_mask,
        vision_mat,
        text_mat,
        all_scene_tokens,
        scene_tokens,
    logit_scale,
    layer_features=None):
    """Dump retrieval-rank and embedding-similarity diagnostics."""
    dump_dir = os.path.dirname(dump_path)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    pos_scores = sim[positive_mask]
    neg_scores = sim[~positive_mask]

    neg_for_max = sim.masked_fill(positive_mask, torch.finfo(sim.dtype).min)
    max_neg_scores, max_neg_idx = neg_for_max.max(dim=1)
    pos_for_row = sim.masked_fill(~positive_mask, torch.finfo(sim.dtype).min).max(dim=1).values
    margins = pos_for_row - max_neg_scores

    sorted_idx = sim.argsort(dim=1, descending=True)
    ranks = []
    top1_idx = []
    for row_idx in range(sim.size(0)):
        positives = torch.nonzero(positive_mask[row_idx], as_tuple=False).view(-1)
        if positives.numel() == 0:
            continue
        row_sorted = sorted_idx[row_idx]
        pos_rank = min(
            int((row_sorted == int(pos_idx.item())).nonzero(as_tuple=False)[0].item()) + 1
            for pos_idx in positives
        )
        ranks.append(pos_rank)
        top1_idx.append(int(row_sorted[0].item()))
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)

    text_text = text_mat @ text_mat.t()
    vision_vision = vision_mat @ vision_mat.t()
    text_offdiag = _offdiag_values(text_text)
    vision_offdiag = _offdiag_values(vision_vision)

    worst_order = torch.argsort(margins)[:20]
    worst_cases = []
    for row_idx in worst_order.tolist():
        if row_idx >= len(all_scene_tokens):
            continue
        gt_indices = torch.nonzero(positive_mask[row_idx], as_tuple=False).view(-1)
        if gt_indices.numel() == 0:
            continue
        gt_idx = int(gt_indices[0].item())
        pred_idx = int(sorted_idx[row_idx, 0].item())
        worst_cases.append({
            'query_index': int(row_idx),
            'gt_scene_token': all_scene_tokens[row_idx],
            'pred_scene_token': scene_tokens[pred_idx],
            'gt_score': float(sim[row_idx, gt_idx].item()),
            'pred_score': float(sim[row_idx, pred_idx].item()),
            'max_neg_scene_token': scene_tokens[int(max_neg_idx[row_idx].item())],
            'max_neg_score': float(max_neg_scores[row_idx].item()),
            'margin': float(margins[row_idx].item()),
            'rank': int(rank_tensor[row_idx].item()) if row_idx < rank_tensor.numel() else None,
        })

    top1_scene_idx = sorted_idx[:, 0]
    t2i_sorted_idx = sim.t().argsort(dim=1, descending=True)
    t2i_top1_clip_idx = t2i_sorted_idx[:, 0]
    t2i_top1_scene_tokens = [all_scene_tokens[idx] for idx in t2i_top1_clip_idx.tolist()]

    report = {
        'count': {
            'vision_clips': int(sim.size(0)),
            'text_scenes': int(sim.size(1)),
        },
        'logit_scale': float(logit_scale.item()) if torch.is_tensor(logit_scale) else float(logit_scale),
        'i2t': {
            'rank': _tensor_stats(rank_tensor),
            'positive_similarity': _tensor_stats(pos_scores),
            'negative_similarity': _tensor_stats(neg_scores),
            'max_negative_similarity': _tensor_stats(max_neg_scores),
            'margin': _tensor_stats(margins),
            'margin_positive_ratio': float((margins > 0).float().mean().item()) if margins.numel() else 0.0,
        },
        'text_text': {
            'offdiag_similarity': _tensor_stats(text_offdiag),
        },
        'vision_vision': {
            'offdiag_similarity': _tensor_stats(vision_offdiag),
        },
        'feature_norms': {
            'vision': _tensor_stats(vision_mat.norm(dim=1)),
            'text': _tensor_stats(text_mat.norm(dim=1)),
        },
        'hubness': {
            'i2t_top1_scene_frequency': _frequency_report(top1_scene_idx, scene_tokens),
            'i2t_max_negative_scene_frequency': _frequency_report(max_neg_idx, scene_tokens),
            't2i_top1_scene_frequency': _frequency_report(
                torch.arange(len(t2i_top1_scene_tokens)),
                t2i_top1_scene_tokens,
            ),
        },
        'geometry': {
            'vision': _feature_geometry_report(vision_mat),
            'text': _feature_geometry_report(text_mat),
        },
        'worst_cases': worst_cases,
    }

    if layer_features:
        report['layers'] = {}
        for name, feats in sorted(layer_features.items()):
            if not feats:
                continue
            mat = torch.stack(feats, dim=0).float()
            norm_mat = F.normalize(mat, dim=-1)
            layer_sim = norm_mat @ norm_mat.t()
            report['layers'][name] = {
                'count': int(mat.size(0)),
                'dim': int(mat.size(1)) if mat.dim() == 2 else None,
                'offdiag_similarity': _tensor_stats(_offdiag_values(layer_sim)),
                'feature_norms': _tensor_stats(mat.norm(dim=1)),
                'geometry': _feature_geometry_report(norm_mat),
            }

    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Saved embedding diagnostics: {}'.format(dump_path))


def load_model_weights(model, cfg, args):
    """Load model weights with backward-compatible checkpoint semantics.

    Priority order:
    - explicit base+align
    - base only
    - align + inferred base (positional checkpoint or config load_from)
    - legacy single checkpoint

    Returns:
        dict: Structured report used for optional dump and strict checks.
    """
    report = {
        'mode': None,
        'base_ckpt': None,
        'align_ckpt': None,
        'single_ckpt': None,
        'align_missing_keys': [],
        'align_unexpected_keys': [],
    }

    if args.base_ckpt and args.align_ckpt:
        # Explicit two-stage load: stable and recommended for align-only ckpt.
        report['mode'] = 'base_plus_align'
        report['base_ckpt'] = args.base_ckpt
        report['align_ckpt'] = args.align_ckpt
        load_checkpoint(model, args.base_ckpt, map_location='cpu')
        print('Loaded base checkpoint: {}'.format(args.base_ckpt))
        missing_keys, unexpected_keys = _load_alignment_checkpoint(model, args.align_ckpt)
        report['align_missing_keys'] = list(missing_keys)
        report['align_unexpected_keys'] = list(unexpected_keys)
        return report

    if args.base_ckpt:
        # Base-only validation path, useful for sanity checks.
        report['mode'] = 'base_only'
        report['base_ckpt'] = args.base_ckpt
        load_checkpoint(model, args.base_ckpt, map_location='cpu')
        print('Loaded base checkpoint: {}'.format(args.base_ckpt))
        return report

    if args.align_ckpt:
        # Backward-compatible path: infer base from positional ckpt or config.
        report['mode'] = 'auto_base_plus_align'
        report['align_ckpt'] = args.align_ckpt
        base_ckpt = args.checkpoint or cfg.get('load_from', None)
        if base_ckpt is None:
            raise ValueError(
                'align-ckpt requires a base checkpoint. '
                'Please provide --base-ckpt, or positional checkpoint, '
                'or set load_from in config.'
            )
        report['base_ckpt'] = base_ckpt
        load_checkpoint(model, base_ckpt, map_location='cpu')
        print('Loaded base checkpoint: {}'.format(base_ckpt))
        missing_keys, unexpected_keys = _load_alignment_checkpoint(model, args.align_ckpt)
        report['align_missing_keys'] = list(missing_keys)
        report['align_unexpected_keys'] = list(unexpected_keys)
        return report

    if args.checkpoint:
        # Legacy single-checkpoint behavior.
        report['mode'] = 'single_ckpt'
        report['single_ckpt'] = args.checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        print('Loaded single checkpoint: {}'.format(args.checkpoint))
        return report

    raise ValueError(
        'No checkpoint specified. Provide positional checkpoint, or '
        '--base-ckpt/--align-ckpt.'
    )


def import_plugin(cfg, config_path):
    """Import project plugin module so custom registries are available.

    BEVFormer plugin modules register datasets/models/hooks at import time.
    Without this import, config construction may fail for custom types.
    """
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)

    module_dir = module_dir.split('/')
    module_path = module_dir[0]
    for m in module_dir[1:]:
        module_path = module_path + '.' + m
    importlib.import_module(module_path)


def main():
    """Run retrieval validation and print dataset-level top-1 scores.

    The script collects per-batch outputs from forward_test and averages
    `acc_i2t_top1` / `acc_t2i_top1` over all validation samples.
    """
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg, args.config)

    # Some configs may accidentally carry dataloader-only keys inside val cfg.
    # Remove them before dataset construction to avoid kwargs errors.
    val_cfg = cfg.data.val.copy()
    # Retrieval validation should always use test/eval dataset behavior.
    # Some inherited configs keep val.test_mode=False for training-time hooks,
    # which would route to prepare_train_data and require gt labels.
    val_cfg['test_mode'] = True
    for key in ['samples_per_gpu', 'workers_per_gpu', 'persistent_workers', 'prefetch_factor', 'shuffle']:
        if key in val_cfg:
            val_cfg.pop(key)

    dataset = build_dataset(val_cfg)
    # Validation uses deterministic order (shuffle=False) to keep results
    # reproducible and comparable across checkpoints.
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=cfg.get('seed', 0),
        shuffler_sampler=cfg.data.get('shuffler_sampler', None),
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.dump_embedding_diagnostics:
        model.return_intermediate_feats = True
    # Load report captures how weights were loaded, then optional strict checks
    # can early-fail before expensive GPU inference starts.
    load_report = load_model_weights(model, cfg, args)
    if args.load_report:
        _dump_load_report(load_report, args.load_report)
    _check_load_report_or_fail(args, load_report)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    loss_scores = []
    all_vision_feats = []
    all_scene_tokens = []
    scene_text_feat_map = {}
    layer_vision_feats = {}
    layer_text_feat_map = {}

    with torch.no_grad():
        for data in dataloader:
            # forward_test of this model returns retrieval diagnostics dicts.
            outputs = model(return_loss=False, rescale=True, **data)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for out in outputs:
                if isinstance(out, dict):
                    if 'loss_align' in out:
                        loss_scores.append(float(out['loss_align']))
                    if 'vision_feat' in out and 'text_feat' in out and 'scene_token' in out:
                        all_vision_feats.append(out['vision_feat'].float())
                        scene_token = out.get('scene_token', None)
                        all_scene_tokens.append(scene_token)
                        if scene_token is not None and scene_token not in scene_text_feat_map:
                            scene_text_feat_map[scene_token] = out['text_feat'].float()
                        embedding_layers = out.get('embedding_layers', None)
                        if isinstance(embedding_layers, dict):
                            for name, feat in embedding_layers.items():
                                feat = feat.float()
                                if name.startswith('vision_'):
                                    layer_vision_feats.setdefault(name, []).append(feat)
                                elif name.startswith('text_') and scene_token is not None:
                                    layer_text_feat_map.setdefault(name, {})
                                    if scene_token not in layer_text_feat_map[name]:
                                        layer_text_feat_map[name][scene_token] = feat

    def _recall_at_k(sim_matrix, positive_mask, k):
        # topk runs on candidate axis(dim=1), so k must be capped by #candidates.
        n_candidates = sim_matrix.size(1)
        k = min(k, n_candidates)
        if k <= 0:
            return 0.0
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return 0.0
        sim_valid = sim_matrix[valid_rows]
        pos_valid = positive_mask[valid_rows]
        topk_idx = sim_valid.topk(k, dim=1, largest=True).indices
        topk_pos = pos_valid.gather(1, topk_idx)
        return topk_pos.any(dim=1).float().mean().item()

    def _multi_positive_infonce(logits, positive_mask):
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return 0.0
        logits = logits[valid_rows]
        positive_mask = positive_mask[valid_rows]
        neg_inf = torch.finfo(logits.dtype).min
        pos_logits = logits.masked_fill(~positive_mask, neg_inf)
        pos_lse = torch.logsumexp(pos_logits, dim=1)
        all_lse = torch.logsumexp(logits, dim=1)
        return -(pos_lse - all_lse).mean().item()

    # Scene-level text deduplicated evaluation:
    # visual side keeps all clips; text side keeps one embedding per scene.
    if len(all_vision_feats) > 0 and len(scene_text_feat_map) > 0:
        vision_mat = torch.stack(all_vision_feats, dim=0)
        scene_tokens = list(scene_text_feat_map.keys())
        text_mat = torch.stack([scene_text_feat_map[tok] for tok in scene_tokens], dim=0)
        vision_mat = F.normalize(vision_mat, dim=-1)
        text_mat = F.normalize(text_mat, dim=-1)
        sim = vision_mat @ text_mat.t()

        n_clip, n_scene = sim.size(0), sim.size(1)
        token_to_scene_idx = {tok: j for j, tok in enumerate(scene_tokens)}
        positive_mask = torch.zeros((n_clip, n_scene), dtype=torch.bool)
        for i, tok in enumerate(all_scene_tokens):
            if tok in token_to_scene_idx:
                positive_mask[i, token_to_scene_idx[tok]] = True

        if positive_mask.sum().item() == 0:
            print('No valid scene-token matches found for scene-level retrieval metrics.')
            return

        logit_scale = model.module.logit_scale.exp().detach().cpu().float()
        logits = sim * logit_scale
        loss_i2t = _multi_positive_infonce(logits, positive_mask)
        loss_t2i = _multi_positive_infonce(logits.t(), positive_mask.t())
        val_loss_align = 0.5 * (loss_i2t + loss_t2i)

        i2t_r1 = _recall_at_k(sim, positive_mask, 1)
        i2t_r5 = _recall_at_k(sim, positive_mask, 5)
        i2t_r10 = _recall_at_k(sim, positive_mask, 10)
        t2i_r1 = _recall_at_k(sim.t(), positive_mask.t(), 1)
        t2i_r5 = _recall_at_k(sim.t(), positive_mask.t(), 5)
        t2i_r10 = _recall_at_k(sim.t(), positive_mask.t(), 10)

        clip_positive_counts = positive_mask.sum(dim=1)
        text_positive_counts = positive_mask.sum(dim=0)

        print(f'val_loss_align: {val_loss_align:.6f}')

        # Keep compatibility keys for caller scripts: top1 lines now mean
        # global R@1 on full validation set.
        print(f'i2t_top1: {i2t_r1:.6f}')
        print(f't2i_top1: {t2i_r1:.6f}')
        print(f'i2t_r5: {i2t_r5:.6f}')
        print(f'i2t_r10: {i2t_r10:.6f}')
        print(f't2i_r5: {t2i_r5:.6f}')
        print(f't2i_r10: {t2i_r10:.6f}')
        # Explicit aliases for the current scene-level relaxed definition.
        # A retrieval is counted correct when the predicted item shares the
        # same scene_token as the query. With one clip per scene, these values
        # are identical to the compatibility metrics above.
        print(f'scene_relaxed_i2t_top1: {i2t_r1:.6f}')
        print(f'scene_relaxed_t2i_top1: {t2i_r1:.6f}')
        print(f'scene_relaxed_i2t_r5: {i2t_r5:.6f}')
        print(f'scene_relaxed_i2t_r10: {i2t_r10:.6f}')
        print(f'scene_relaxed_t2i_r5: {t2i_r5:.6f}')
        print(f'scene_relaxed_t2i_r10: {t2i_r10:.6f}')
        print(
            'scene_relaxed_positive_counts: '
            f'clip_min={int(clip_positive_counts.min().item())}, '
            f'clip_max={int(clip_positive_counts.max().item())}, '
            f'text_min={int(text_positive_counts.min().item())}, '
            f'text_max={int(text_positive_counts.max().item())}'
        )
        if args.dump_embedding_diagnostics:
            layer_features = {}
            layer_features.update(layer_vision_feats)
            for name, feat_map in layer_text_feat_map.items():
                layer_features[name] = [feat_map[tok] for tok in scene_tokens if tok in feat_map]
            _dump_embedding_diagnostics(
                args.dump_embedding_diagnostics,
                sim,
                positive_mask,
                vision_mat,
                text_mat,
                all_scene_tokens,
                scene_tokens,
                logit_scale,
                layer_features=layer_features,
            )
        return

    print('No scene-level retrieval metrics were produced by the model outputs.')


if __name__ == '__main__':
    main()

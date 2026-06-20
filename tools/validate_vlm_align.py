import argparse
import importlib
import json
import os

import torch
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
    # Load report captures how weights were loaded, then optional strict checks
    # can early-fail before expensive GPU inference starts.
    load_report = load_model_weights(model, cfg, args)
    if args.load_report:
        _dump_load_report(load_report, args.load_report)
    _check_load_report_or_fail(args, load_report)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    i2t_scores = []
    t2i_scores = []
    loss_scores = []

    with torch.no_grad():
        for data in dataloader:
            # forward_test of this model returns retrieval diagnostics dicts.
            outputs = model(return_loss=False, rescale=True, **data)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for out in outputs:
                if isinstance(out, dict):
                    if 'acc_i2t_top1' in out:
                        i2t_scores.append(float(out['acc_i2t_top1']))
                    if 'acc_t2i_top1' in out:
                        t2i_scores.append(float(out['acc_t2i_top1']))
                    if 'loss_align' in out:
                        loss_scores.append(float(out['loss_align']))

    if len(i2t_scores) == 0 or len(t2i_scores) == 0:
        # Usually indicates mismatched model/output path or missing retrieval
        # fields in inference outputs.
        print('No retrieval metrics were produced by the model outputs.')
        return

    # Final metrics are simple means across the collected per-sample values.
    i2t_top1 = sum(i2t_scores) / len(i2t_scores)
    t2i_top1 = sum(t2i_scores) / len(t2i_scores)
    if len(loss_scores) > 0:
        val_loss_align = sum(loss_scores) / len(loss_scores)
        print(f'val_loss_align: {val_loss_align:.6f}')
    print(f'i2t_top1: {i2t_top1:.6f}')
    print(f't2i_top1: {t2i_top1:.6f}')


if __name__ == '__main__':
    main()

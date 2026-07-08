#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run alignment training and auto-validate each saved align checkpoint.'
    )
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', default=None, help='work dir for training outputs')
    parser.add_argument('--base-ckpt', default=None, help='base checkpoint path for validation')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='validation batch size per gpu')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='validation workers per gpu')
    parser.add_argument(
        '--val-cuda-visible-devices',
        default=None,
        help='optional CUDA_VISIBLE_DEVICES value used only for validation subprocess')
    parser.add_argument('--poll-seconds', type=float, default=10.0, help='checkpoint polling interval in seconds')
    parser.add_argument('--summary-file', default='align_val_summary.json', help='summary json file name in work dir')
    parser.add_argument('--load-report-dir', default='align_val_load_reports', help='directory name in work dir for per-epoch load reports')
    parser.add_argument(
        '--dump-embedding-diagnostics',
        action='store_true',
        help='dump per-epoch embedding diagnostics json during validation')
    parser.add_argument(
        '--embedding-diagnostics-dir',
        default='embedding_diagnostics',
        help='directory name in work dir for per-epoch embedding diagnostics json')
    parser.add_argument(
        '--embedding-diagnostics-name-template',
        default='embedding_diag_epoch_{epoch}.json',
        help='output filename template inside embedding-diagnostics-dir')
    parser.add_argument(
        '--embedding-diagnostics-interval',
        type=int,
        default=1,
        help='run embedding diagnostics every N validated epochs when enabled')
    parser.add_argument(
        '--diagnostics-ann-file',
        default=None,
        help='optional ann_file override used only for diagnostics export')
    parser.add_argument(
        '--diagnostics-scene-json',
        default=None,
        help='optional scene json override used only for diagnostics export')
    parser.add_argument(
        '--diagnostics-offline-meta-only',
        action='store_true',
        help='set data.val.offline_meta_only=True for diagnostics export')
    parser.add_argument(
        '--diagnostics-subset-one-per-scene',
        action='store_true',
        help='build and reuse a fixed subset ann_file containing one sample per scene')
    parser.add_argument(
        '--diagnostics-subset-path',
        default=None,
        help='output path for the generated one-per-scene subset pkl')
    parser.add_argument('--max-align-missing-keys', type=int, default=None, help='pass-through to validate_vlm_align.py')
    parser.add_argument('--fail-on-unexpected-keys', action='store_true', help='pass-through to validate_vlm_align.py')
    parser.add_argument('--stop-on-val-fail', action='store_true', help='stop training immediately if any epoch validation fails')
    parser.add_argument(
        '--validate-after-train',
        action='store_true',
        help='run validations only after training exits (no concurrent train/val)')
    args, train_extra = parser.parse_known_args()
    # Keep compatibility with shell-style separator:
    #   train_validate_vlm_align.py ... -- --cfg-options ...
    # argparse keeps the separator token in unknown args, but train.py does not
    # accept a standalone "--", so strip it before forwarding.
    if train_extra and train_extra[0] == '--':
        train_extra = train_extra[1:]
    args.train_extra = train_extra
    return args


def infer_work_dir(cfg, config_path, cli_work_dir):
    if cli_work_dir:
        return Path(cli_work_dir)
    cfg_work_dir = cfg.get('work_dir', None)
    if cfg_work_dir:
        return Path(cfg_work_dir)
    return Path('./work_dirs') / Path(config_path).stem


def extract_epoch_from_name(path: Path):
    m = re.search(r'align_trainable_epoch_(\d+)\.pth$', path.name)
    if m:
        return int(m.group(1))
    return -1


def parse_metrics(stdout_text):
    val_loss_align = None
    i2t = None
    t2i = None
    for line in stdout_text.splitlines():
        if line.startswith('val_loss_align:'):
            val_loss_align = float(line.split(':', 1)[1].strip())
        if line.startswith('i2t_top1:'):
            i2t = float(line.split(':', 1)[1].strip())
        if line.startswith('t2i_top1:'):
            t2i = float(line.split(':', 1)[1].strip())
    return val_loss_align, i2t, t2i


def save_summary(summary_path: Path, records):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def ensure_one_per_scene_subset(source_ann: Path, subset_ann: Path, clip_length: int):
    subset_ann.parent.mkdir(parents=True, exist_ok=True)
    if subset_ann.exists():
        return subset_ann

    clip_length = max(int(clip_length), 1)

    with source_ann.open('rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
        metadata = data.get('metadata', {})
    elif isinstance(data, list):
        infos = data
        metadata = {}
    else:
        raise ValueError('Unsupported ann_file structure for subset generation: {}'.format(type(data).__name__))

    by_scene = defaultdict(list)
    for info in infos:
        scene_token = info.get('scene_token', None)
        if scene_token is None:
            continue
        by_scene[scene_token].append(info)

    subset_infos = []
    for scene_token, scene_infos in by_scene.items():
        scene_infos.sort(key=lambda info: int(info.get('frame_idx', 0)))
        subset_infos.extend(scene_infos[:clip_length])

    subset_data = {'infos': subset_infos, 'metadata': metadata}
    with subset_ann.open('wb') as f:
        pickle.dump(subset_data, f)

    print(
        'Saved diagnostics subset: {} (samples={}, unique_scenes={}, clip_length={})'.format(
            subset_ann,
            len(subset_infos),
            len(by_scene),
            clip_length,
        )
    )
    return subset_ann


def resolve_diagnostics_ann_file(args, cfg, work_dir: Path):
    ann_file = args.diagnostics_ann_file
    if ann_file is None:
        ann_file = cfg.data.val.get('ann_file', None)
    if ann_file is None:
        raise ValueError('diagnostics ann_file is required: use --diagnostics-ann-file or set data.val.ann_file in config.')

    ann_path = Path(ann_file)
    if args.diagnostics_subset_one_per_scene:
        subset_path = args.diagnostics_subset_path
        if subset_path is None:
            subset_path = work_dir / 'tmp_val_one_per_scene.pkl'
        subset_path = Path(subset_path)
        frames = cfg.data.val.get('frames', None)
        clip_length = len(frames) if frames is not None else 40
        return ensure_one_per_scene_subset(ann_path, subset_path, clip_length)

    return ann_path


def build_validate_cfg_overrides(args, cfg, diagnostics_ann_file=None, force_offline_meta_only=False):
    overrides = []
    train_extra = list(getattr(args, 'train_extra', []) or [])
    offline_train_mode = any(x.startswith('model.run_mode=offline_train') for x in train_extra)
    if offline_train_mode:
        overrides.extend([
            'model.run_mode=offline_infer_validate',
            'model.offline_split=val',
            'data.val.offline_meta_only=True',
        ])
        if force_offline_meta_only:
            overrides.append('data.val.offline_meta_only=True')

    scene_json_override = args.diagnostics_scene_json
    if scene_json_override is None:
        scene_json_override = next((x.split('=', 1)[1] for x in train_extra if x.startswith('model.scene_json=')), None)
    if scene_json_override is not None:
        overrides.append('model.scene_json={}'.format(scene_json_override))

    if diagnostics_ann_file is not None:
        overrides.append('data.val.ann_file={}'.format(diagnostics_ann_file))

    return overrides


def should_run_embedding_diagnostics(args, epoch: int):
    return args.dump_embedding_diagnostics and args.embedding_diagnostics_interval > 0 and epoch % args.embedding_diagnostics_interval == 0


def _run_validate_subprocess(
    args,
    config_path,
    base_ckpt,
    align_ckpt,
    work_dir: Path,
    *,
    load_report_dir_name,
    diagnostics_path=None,
    diagnostics_ann_file=None,
    force_offline_meta_only=False,
):
    epoch = extract_epoch_from_name(align_ckpt)
    cmd = [
        sys.executable,
        'tools/validate_vlm_align.py',
        config_path,
        '--base-ckpt',
        str(base_ckpt),
        '--align-ckpt',
        str(align_ckpt),
        '--samples-per-gpu',
        str(args.samples_per_gpu),
        '--workers-per-gpu',
        str(args.workers_per_gpu),
    ]

    cfg_overrides = build_validate_cfg_overrides(
        args,
        Config.fromfile(config_path),
        diagnostics_ann_file,
        force_offline_meta_only=force_offline_meta_only,
    )
    if cfg_overrides:
        cmd.append('--cfg-options')
        cmd.extend(cfg_overrides)

    if args.max_align_missing_keys is not None:
        cmd.extend(['--max-align-missing-keys', str(args.max_align_missing_keys)])
    if args.fail_on_unexpected_keys:
        cmd.append('--fail-on-unexpected-keys')

    load_report_dir = work_dir / load_report_dir_name
    load_report_dir.mkdir(parents=True, exist_ok=True)
    load_report_path = load_report_dir / (align_ckpt.stem + '.json')
    cmd.extend(['--load-report', str(load_report_path)])

    if diagnostics_path is not None:
        cmd.extend(['--dump-embedding-diagnostics', str(diagnostics_path)])

    env = os.environ.copy()
    if args.val_cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.val_cuda_visible_devices)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return load_report_path, diagnostics_path, proc


def run_validate(args, config_path, base_ckpt, align_ckpt, work_dir: Path):
    load_report_path, diagnostics_path, proc = _run_validate_subprocess(
        args,
        config_path,
        base_ckpt,
        align_ckpt,
        work_dir,
        load_report_dir_name=args.load_report_dir,
    )
    val_loss_align, i2t, t2i = parse_metrics(proc.stdout)

    return {
        'epoch': extract_epoch_from_name(align_ckpt),
        'align_ckpt': str(align_ckpt),
        'load_report': str(load_report_path),
        'embedding_diagnostics': str(diagnostics_path) if diagnostics_path is not None else None,
        'returncode': proc.returncode,
        'val_loss_align': val_loss_align,
        'i2t_top1': i2t,
        't2i_top1': t2i,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }


def run_embedding_diagnostics(args, config_path, base_ckpt, align_ckpt, work_dir: Path):
    epoch = extract_epoch_from_name(align_ckpt)
    cfg = Config.fromfile(config_path)
    diagnostics_ann_file = resolve_diagnostics_ann_file(args, cfg, work_dir)

    diagnostics_dir = work_dir / args.embedding_diagnostics_dir
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_name = args.embedding_diagnostics_name_template.format(epoch=epoch)
    diagnostics_path = diagnostics_dir / diagnostics_name

    load_report_path, _, proc = _run_validate_subprocess(
        args,
        config_path,
        base_ckpt,
        align_ckpt,
        work_dir,
        load_report_dir_name='embedding_diagnostics_load_reports',
        diagnostics_path=diagnostics_path,
        diagnostics_ann_file=diagnostics_ann_file,
        force_offline_meta_only=args.diagnostics_offline_meta_only,
    )

    return {
        'embedding_diagnostics': str(diagnostics_path),
        'diagnostics_load_report': str(load_report_path),
        'diagnostics_returncode': proc.returncode,
        'diagnostics_stdout': proc.stdout,
        'diagnostics_stderr': proc.stderr,
    }


def record_failed(rec):
    return rec['returncode'] != 0 or rec.get('diagnostics_returncode', 0) != 0


def main():
    args = parse_args()
    config_path = args.config

    cfg = Config.fromfile(config_path)
    work_dir = infer_work_dir(cfg, config_path, args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_embedding_diagnostics and args.embedding_diagnostics_interval < 1:
        raise ValueError('--embedding-diagnostics-interval must be >= 1 when diagnostics export is enabled.')

    base_ckpt = args.base_ckpt or cfg.get('load_from', None)
    if base_ckpt is None:
        raise ValueError('base checkpoint is required: use --base-ckpt or set load_from in config.')

    train_cmd = [
        sys.executable,
        'tools/train.py',
        config_path,
        '--work-dir',
        str(work_dir),
        '--no-validate',
    ]
    if args.train_extra:
        train_cmd.extend(args.train_extra)

    print('Launch train command: {}'.format(' '.join(train_cmd)))
    train_proc = subprocess.Popen(train_cmd)

    seen = set()
    records = []
    summary_path = work_dir / args.summary_file

    try:
        while True:
            ckpts = sorted(
                work_dir.glob('align_trainable_epoch_*.pth'),
                key=extract_epoch_from_name,
            )

            if not args.validate_after_train:
                for ckpt in ckpts:
                    ckpt_s = str(ckpt)
                    if ckpt_s in seen:
                        continue

                    print('Validate new checkpoint: {}'.format(ckpt_s))
                    rec = run_validate(args, config_path, base_ckpt, ckpt, work_dir)
                    if should_run_embedding_diagnostics(args, rec['epoch']):
                        rec.update(run_embedding_diagnostics(args, config_path, base_ckpt, ckpt, work_dir))
                    records.append(rec)
                    seen.add(ckpt_s)
                    save_summary(summary_path, records)

                    if record_failed(rec):
                        print('Validation failed on {} with code {}'.format(ckpt_s, rec['returncode']))
                        if rec.get('stderr'):
                            print('Validation stderr (tail):')
                            print('\n'.join(rec['stderr'].splitlines()[-30:]))
                        elif rec.get('stdout'):
                            print('Validation stdout (tail):')
                            print('\n'.join(rec['stdout'].splitlines()[-30:]))
                        if rec.get('diagnostics_returncode', 0) != 0:
                            print('Diagnostics failed on {} with code {}'.format(ckpt_s, rec['diagnostics_returncode']))
                            if rec.get('diagnostics_stderr'):
                                print('Diagnostics stderr (tail):')
                                print('\n'.join(rec['diagnostics_stderr'].splitlines()[-30:]))
                            elif rec.get('diagnostics_stdout'):
                                print('Diagnostics stdout (tail):')
                                print('\n'.join(rec['diagnostics_stdout'].splitlines()[-30:]))
                        if args.stop_on_val_fail and train_proc.poll() is None:
                            train_proc.terminate()

            train_rc = train_proc.poll()
            if train_rc is not None:
                # Final sweep in case a checkpoint appears right before exit.
                ckpts = sorted(
                    work_dir.glob('align_trainable_epoch_*.pth'),
                    key=extract_epoch_from_name,
                )
                for ckpt in ckpts:
                    ckpt_s = str(ckpt)
                    if ckpt_s in seen:
                        continue
                    print('Validate late checkpoint: {}'.format(ckpt_s))
                    rec = run_validate(args, config_path, base_ckpt, ckpt, work_dir)
                    if should_run_embedding_diagnostics(args, rec['epoch']):
                        rec.update(run_embedding_diagnostics(args, config_path, base_ckpt, ckpt, work_dir))
                    records.append(rec)
                    seen.add(ckpt_s)
                    save_summary(summary_path, records)
                break

            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        if train_proc.poll() is None:
            train_proc.terminate()

    failed_val = [r for r in records if record_failed(r)]
    print('Validation summary saved to: {}'.format(summary_path))

    if train_proc.returncode not in (0, None):
        print('Training failed with code {}'.format(train_proc.returncode))
        sys.exit(train_proc.returncode)

    if failed_val:
        print('There are {} failed validation runs.'.format(len(failed_val)))
        sys.exit(1)


if __name__ == '__main__':
    main()

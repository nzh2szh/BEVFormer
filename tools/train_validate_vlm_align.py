#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
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


def run_validate(args, config_path, base_ckpt, align_ckpt, work_dir: Path):
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

    train_extra = list(getattr(args, 'train_extra', []) or [])
    offline_train_mode = any(x.startswith('model.run_mode=offline_train') for x in train_extra)
    if offline_train_mode:
        cmd.extend([
            '--cfg-options',
            'model.run_mode=offline_infer_validate',
            'model.offline_split=val',
            'data.val.offline_meta_only=True',
        ])
        scene_json_override = next((x for x in train_extra if x.startswith('model.scene_json=')), None)
        if scene_json_override is not None:
            cmd.append(scene_json_override)

    if args.max_align_missing_keys is not None:
        cmd.extend(['--max-align-missing-keys', str(args.max_align_missing_keys)])
    if args.fail_on_unexpected_keys:
        cmd.append('--fail-on-unexpected-keys')

    load_report_dir = work_dir / args.load_report_dir
    load_report_dir.mkdir(parents=True, exist_ok=True)
    load_report_path = load_report_dir / (align_ckpt.stem + '.json')
    cmd.extend(['--load-report', str(load_report_path)])

    env = os.environ.copy()
    if args.val_cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.val_cuda_visible_devices)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    val_loss_align, i2t, t2i = parse_metrics(proc.stdout)

    return {
        'epoch': extract_epoch_from_name(align_ckpt),
        'align_ckpt': str(align_ckpt),
        'load_report': str(load_report_path),
        'returncode': proc.returncode,
        'val_loss_align': val_loss_align,
        'i2t_top1': i2t,
        't2i_top1': t2i,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }


def main():
    args = parse_args()
    config_path = args.config

    cfg = Config.fromfile(config_path)
    work_dir = infer_work_dir(cfg, config_path, args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

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
                    records.append(rec)
                    seen.add(ckpt_s)
                    save_summary(summary_path, records)

                    if rec['returncode'] != 0:
                        print('Validation failed on {} with code {}'.format(ckpt_s, rec['returncode']))
                        if rec.get('stderr'):
                            print('Validation stderr (tail):')
                            print('\n'.join(rec['stderr'].splitlines()[-30:]))
                        elif rec.get('stdout'):
                            print('Validation stdout (tail):')
                            print('\n'.join(rec['stdout'].splitlines()[-30:]))
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
                    records.append(rec)
                    seen.add(ckpt_s)
                    save_summary(summary_path, records)
                break

            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        if train_proc.poll() is None:
            train_proc.terminate()

    failed_val = [r for r in records if r['returncode'] != 0]
    print('Validation summary saved to: {}'.format(summary_path))

    if train_proc.returncode not in (0, None):
        print('Training failed with code {}'.format(train_proc.returncode))
        sys.exit(train_proc.returncode)

    if failed_val:
        print('There are {} failed validation runs.'.format(len(failed_val)))
        sys.exit(1)


if __name__ == '__main__':
    main()

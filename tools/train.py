# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def _boot_debug(msg):
    if os.environ.get('DEBUG_TRAIN_BOOT', '0') == '1':
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        rank = os.environ.get('RANK', '?')
        local_rank = os.environ.get('LOCAL_RANK', '?')
        print(f'[BOOT][{ts}][rank={rank}][local_rank={local_rank}] {msg}', flush=True)


def _resolve_extract_mode(cfg):
    model_cfg = cfg.get('model', {})
    if not isinstance(model_cfg, dict):
        return None, None
    run_mode = model_cfg.get('run_mode', None)
    if run_mode is None:
        return None, None

    mode_alias = {
        'origin': 'online',
        'offline_extract_bev': 'extract',
    }
    resolved_mode = mode_alias.get(run_mode, run_mode)
    split = model_cfg.get('offline_split', None)
    if split is None:
        split = 'train'
    return resolved_mode, split


def _strip_dataloader_keys(dataset_cfg):
    """Recursively remove dataloader-only keys before build_dataset."""
    dataloader_only_keys = {
        'samples_per_gpu',
        'workers_per_gpu',
        'persistent_workers',
        'num_gpus',
        'dist',
        'shuffle',
        'seed',
        'drop_last',
        'pin_memory',
        'prefetch_factor',
        'timeout',
        'sampler',
        'batch_sampler',
    }

    if isinstance(dataset_cfg, list):
        return [_strip_dataloader_keys(x) for x in dataset_cfg]
    if not isinstance(dataset_cfg, dict):
        return dataset_cfg

    sanitized = {}
    for key, val in dataset_cfg.items():
        if key in dataloader_only_keys:
            continue
        sanitized[key] = _strip_dataloader_keys(val)
    return sanitized


def _set_dataset_flag(dataset_cfg, key, value):
    """Recursively set dataset flags for plain/Concat/Repeat wrappers."""
    if isinstance(dataset_cfg, list):
        for item in dataset_cfg:
            _set_dataset_flag(item, key, value)
        return
    if not isinstance(dataset_cfg, dict):
        return

    dataset_cfg[key] = value
    if isinstance(dataset_cfg.get('dataset', None), dict):
        _set_dataset_flag(dataset_cfg['dataset'], key, value)
    if isinstance(dataset_cfg.get('datasets', None), list):
        _set_dataset_flag(dataset_cfg['datasets'], key, value)


def _sanitize_lr_config(cfg):
    """Normalize lr_config overrides that MMCV hooks do not accept together."""
    lr_cfg = cfg.get('lr_config', None)
    if not isinstance(lr_cfg, dict):
        return

    if lr_cfg.get('policy', None) == 'Fixed':
        lr_cfg.pop('min_lr_ratio', None)
        lr_cfg.pop('min_lr', None)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    _boot_debug('parse_args done')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _sanitize_lr_config(cfg)
    _boot_debug('config loaded and merged')
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    _boot_debug('custom imports done')

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
    _boot_debug('plugin import done')
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        _boot_debug(f'init_dist start launcher={args.launcher}')
        init_dist(args.launcher, **cfg.dist_params)
        _boot_debug('init_dist done')
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    _boot_debug('build_model done')
    model.init_weights()
    _boot_debug('model.init_weights done')

    logger.info(f'Model:\n{model}')

    resolved_mode, offline_split = _resolve_extract_mode(cfg)
    if resolved_mode == 'extract':
        if offline_split not in {'train', 'val', 'test'}:
            raise ValueError(
                'Invalid model.offline_split={} in extract mode; expected train/val/test.'.format(
                    offline_split)
            )

        split_cfg = cfg.data.get(offline_split, None)
        if split_cfg is None:
            raise KeyError(
                'cfg.data.{} is required in extract mode when model.offline_split={}.'.format(
                    offline_split, offline_split)
            )

        cfg.data.train = _strip_dataloader_keys(split_cfg)

        val_cfg = cfg.data.get('val', None)
        val_pipeline = None
        if isinstance(val_cfg, dict):
            val_pipeline = val_cfg.get('pipeline', None)

        if val_pipeline is not None:
            # Export path should be deterministic; always use eval-style pipeline.
            cfg.data.train['pipeline'] = copy.deepcopy(val_pipeline)
        elif isinstance(cfg.data.train, dict) and cfg.data.train.get('pipeline', None) is not None:
            # Fallback: keep split pipeline when val pipeline is unavailable.
            cfg.data.train['pipeline'] = copy.deepcopy(cfg.data.train['pipeline'])

        if isinstance(cfg.data.train, dict):
            cfg.data.train['test_mode'] = False
            # Extract mode may use eval-style pipelines without gt_labels_3d.
            # Disable empty-gt filtering to avoid KeyError in prepare_train_data.
            _set_dataset_flag(cfg.data.train, 'filter_empty_gt', False)

        logger.info(
            'Extract mode detected: remap data.train to cfg.data.%s and force eval pipeline for offline BEV dump.',
            offline_split,
        )
        train_ann_file = cfg.data.train.get('ann_file', '<missing ann_file>')
        model_scene_json = cfg.model.get('scene_json', '<missing scene_json>')
        train_pipeline = cfg.data.train.get('pipeline', [])
        if isinstance(train_pipeline, list):
            pipeline_types = [p.get('type', 'Unknown') for p in train_pipeline if isinstance(p, dict)]
        else:
            pipeline_types = ['<non-list pipeline>']
        logger.info(
            'Offline extract source summary: split=%s, ann_file=%s, scene_json=%s, pipeline=%s',
            offline_split,
            train_ann_file,
            model_scene_json,
            ' -> '.join(pipeline_types),
        )

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

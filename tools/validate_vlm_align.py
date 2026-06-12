import argparse
import importlib
import os

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Validate BEVFormer VLM alignment model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='val batch size per gpu')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='num workers per gpu')
    return parser.parse_args()


def import_plugin(cfg, config_path):
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
    args = parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)

    dataset = build_dataset(cfg.data.val)
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
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    i2t_scores = []
    t2i_scores = []

    with torch.no_grad():
        for data in dataloader:
            outputs = model(return_loss=False, rescale=True, **data)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for out in outputs:
                if isinstance(out, dict):
                    if 'acc_i2t_top1' in out:
                        i2t_scores.append(float(out['acc_i2t_top1']))
                    if 'acc_t2i_top1' in out:
                        t2i_scores.append(float(out['acc_t2i_top1']))

    if len(i2t_scores) == 0 or len(t2i_scores) == 0:
        print('No retrieval metrics were produced by the model outputs.')
        return

    i2t_top1 = sum(i2t_scores) / len(i2t_scores)
    t2i_top1 = sum(t2i_scores) / len(t2i_scores)
    print(f'i2t_top1: {i2t_top1:.6f}')
    print(f't2i_top1: {t2i_top1:.6f}')


if __name__ == '__main__':
    main()

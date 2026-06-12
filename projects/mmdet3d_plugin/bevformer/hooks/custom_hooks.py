from mmcv.runner.hooks.hook import HOOKS, Hook
import os

import torch
from mmcv.parallel import is_module_wrapper
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class SaveTrainableStateDictHook(Hook):
    """Save model.trainable_state_dict() instead of full checkpoints."""

    def __init__(self, interval=1, by_epoch=True, out_dir=None, filename_tmpl=None):
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = out_dir
        self.filename_tmpl = filename_tmpl

    def _get_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def _save(self, runner):
        if getattr(runner, 'rank', 0) != 0:
            return

        model = self._get_model(runner)
        if not hasattr(model, 'trainable_state_dict'):
            raise AttributeError('Model does not implement trainable_state_dict().')

        out_dir = self.out_dir if self.out_dir else runner.work_dir
        os.makedirs(out_dir, exist_ok=True)

        if self.by_epoch:
            filename_tmpl = self.filename_tmpl or 'align_trainable_epoch_{}.pth'
            filename = filename_tmpl.format(runner.epoch + 1)
        else:
            filename_tmpl = self.filename_tmpl or 'align_trainable_iter_{}.pth'
            filename = filename_tmpl.format(runner.iter + 1)

        filepath = os.path.join(out_dir, filename)
        payload = {
            'meta': {
                'epoch': runner.epoch + 1,
                'iter': runner.iter + 1,
            },
            'state_dict': model.trainable_state_dict(),
        }
        torch.save(payload, filepath)
        runner.logger.info('Saved trainable alignment weights to {}'.format(filepath))

    def after_train_epoch(self, runner):
        if self.by_epoch and self.every_n_epochs(runner, self.interval):
            self._save(runner)

    def after_train_iter(self, runner):
        if (not self.by_epoch) and self.every_n_iters(runner, self.interval):
            self._save(runner)

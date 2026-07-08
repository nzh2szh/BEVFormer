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


@HOOKS.register_module()
class StopAfterTargetEpochHook(Hook):
    """Stop an epoch-based run right after a target human epoch finishes."""

    def __init__(self, target_epoch):
        self.target_epoch = int(target_epoch)

    def after_train_epoch(self, runner):
        completed_epoch = runner.epoch + 1
        if completed_epoch < self.target_epoch:
            return

        current_limit = getattr(runner, '_max_epochs', completed_epoch)
        new_limit = min(int(current_limit), completed_epoch)
        runner._max_epochs = new_limit
        if hasattr(runner, 'logger'):
            runner.logger.info(
                'StopAfterTargetEpochHook: completed epoch %d, clamp max_epochs to %d.'
                % (completed_epoch, new_limit))


@HOOKS.register_module()
class DebugTrainableUpdateHook(Hook):
    """Periodically log lr/grad/update stats of key trainable parameters."""

    def __init__(self, interval=200, param_keywords=None, max_params=20, log_detail=True, log_summary=True):
        self.interval = int(interval)
        self.param_keywords = list(param_keywords) if param_keywords is not None else [
            'vision_projector',
            'text_projector',
            'logit_scale',
            'temporal_encoder',
        ]
        self.max_params = int(max_params)
        self.log_detail = bool(log_detail)
        self.log_summary = bool(log_summary)
        self._tracked = []
        self._summary_tracked = []
        self._init_params = {}

    def _get_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def _match(self, name):
        return any(k in name for k in self.param_keywords)

    def _group_name(self, name):
        for k in self.param_keywords:
            if k in name:
                return k
        return 'other'

    def _resolve_lr(self, runner):
        try:
            cur = runner.current_lr()
            if isinstance(cur, dict):
                vals = []
                for v in cur.values():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        vals.append(float(v[0]))
                if len(vals) > 0:
                    return vals[0]
            if isinstance(cur, (list, tuple)) and len(cur) > 0:
                return float(cur[0])
        except Exception:
            pass

        optimizer = getattr(runner, 'optimizer', None)
        if optimizer is not None and hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            return float(optimizer.param_groups[0].get('lr', 0.0))
        return None

    def before_run(self, runner):
        model = self._get_model(runner)
        tracked = []
        summary_tracked = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not self._match(name):
                continue
            summary_tracked.append((name, param))
            self._init_params[name] = param.detach().float().cpu().clone()
            if len(tracked) < self.max_params:
                tracked.append((name, param))
        self._tracked = tracked
        self._summary_tracked = summary_tracked

        names = [n for n, _ in tracked]
        runner.logger.info(
            'DebugTrainableUpdateHook: detail_tracking={} summary_tracking={} interval={}, detail={}, summary={}, keywords={}. names={}'.format(
                len(names),
                len(summary_tracked),
                self.interval,
                self.log_detail,
                self.log_summary,
                self.param_keywords,
                names,
            )
        )

    def after_train_iter(self, runner):
        if self.interval <= 0:
            return
        if not self.every_n_iters(runner, self.interval):
            return

        lr = self._resolve_lr(runner)
        lr_text = 'NA' if lr is None else '{:.6e}'.format(lr)
        runner.logger.info('[DebugUpdate] iter={} lr={}'.format(runner.iter + 1, lr_text))

        all_grad = []
        all_delta = []
        all_param_abs = []
        group_grad = {}
        group_delta = {}
        group_param_abs = {}
        group_param_count = {}

        for name, param in self._summary_tracked:
            group = self._group_name(name)
            group_param_count[group] = group_param_count.get(group, 0) + 1

            param_abs_mean = float(param.detach().float().abs().mean().item())
            all_param_abs.append(param_abs_mean)
            group_param_abs.setdefault(group, []).append(param_abs_mean)

            grad_norm = None
            if param.grad is not None:
                grad_norm = float(param.grad.detach().float().norm().item())
                all_grad.append(grad_norm)
                group_grad.setdefault(group, []).append(grad_norm)

            init = self._init_params.get(name, None)
            delta_mean_abs = None
            if init is not None:
                cur = param.detach().float().cpu()
                delta_mean_abs = float((cur - init).abs().mean().item())
                all_delta.append(delta_mean_abs)
                group_delta.setdefault(group, []).append(delta_mean_abs)

        for name, param in self._tracked:
            grad_norm = None
            if param.grad is not None:
                grad_norm = float(param.grad.detach().float().norm().item())

            init = self._init_params.get(name, None)
            delta_mean_abs = None
            if init is not None:
                cur = param.detach().float().cpu()
                delta_mean_abs = float((cur - init).abs().mean().item())

            if self.log_detail:
                runner.logger.info(
                    '[DebugUpdate] {} grad_norm={} delta_mean_abs={}'.format(
                        name,
                        'None' if grad_norm is None else '{:.6e}'.format(grad_norm),
                        'None' if delta_mean_abs is None else '{:.6e}'.format(delta_mean_abs),
                    )
                )

        if self.log_summary:
            mean_grad = 0.0 if len(all_grad) == 0 else sum(all_grad) / float(len(all_grad))
            mean_delta = 0.0 if len(all_delta) == 0 else sum(all_delta) / float(len(all_delta))
            mean_param_abs = 0.0 if len(all_param_abs) == 0 else sum(all_param_abs) / float(len(all_param_abs))
            update_ratio = 0.0 if mean_param_abs <= 0.0 else mean_delta / mean_param_abs
            runner.logger.info(
                '[DebugUpdateSummary] tracked={} grad_count={} mean_grad={} delta_count={} mean_delta={} param_count={} mean_param_abs={} update_ratio={}'.format(
                    len(self._summary_tracked),
                    len(all_grad),
                    '{:.6e}'.format(mean_grad),
                    len(all_delta),
                    '{:.6e}'.format(mean_delta),
                    len(all_param_abs),
                    '{:.6e}'.format(mean_param_abs),
                    '{:.6e}'.format(update_ratio),
                )
            )

            for k in self.param_keywords:
                gvals = group_grad.get(k, [])
                dvals = group_delta.get(k, [])
                pvals = group_param_abs.get(k, [])
                gmean = 0.0 if len(gvals) == 0 else sum(gvals) / float(len(gvals))
                dmean = 0.0 if len(dvals) == 0 else sum(dvals) / float(len(dvals))
                pmean = 0.0 if len(pvals) == 0 else sum(pvals) / float(len(pvals))
                uratio = 0.0 if pmean <= 0.0 else dmean / pmean
                runner.logger.info(
                    '[DebugUpdateSummary] group={} param_count={} grad_count={} mean_grad={} delta_count={} mean_delta={} mean_param_abs={} update_ratio={}'.format(
                        k,
                        group_param_count.get(k, 0),
                        len(gvals),
                        '{:.6e}'.format(gmean),
                        len(dvals),
                        '{:.6e}'.format(dmean),
                        '{:.6e}'.format(pmean),
                        '{:.6e}'.format(uratio),
                    )
                )

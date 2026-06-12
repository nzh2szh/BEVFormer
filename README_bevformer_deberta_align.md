date: 202606111600

改动：
新增权重保存 Hook
projects/mmdet3d_plugin/bevformer/hooks/custom_hooks.py
新增 SaveTrainableStateDictHook，按周期调用 model.trainable_state_dict() 并保存为 pth。

导出 Hook 注册
projects/mmdet3d_plugin/bevformer/hooks/init.py

对齐配置接入只存新增权重
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py
设置 checkpoint_config = None，并增加 custom_hooks:

type: SaveTrainableStateDictHook
by_epoch: True
interval: 1
filename_tmpl: align_trainable_epoch_{}.pth
结果：

每个 epoch 会在 work_dir 下生成 align_trainable_epoch_N.pth。
文件里只包含新增可训练部分（空间池化、升维层、时序 Transformer、Text Projector、温度参数等），满足你文档里“只保存新增对齐环节权重”的要求。
我已做静态错误检查，相关文件均无报错。
可直接运行：

训练:
python tools/train.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --no-validate

验证:
python tools/validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py 你的权重路径

--------------------------------------------------------------------------------------------------------

date: 202606111715

完成项核对：

2D Spatial PE 已加
在视觉塔里先给 BEV 稠密特征注入空间位置编码，再做空间 attention pooling。实现位置在 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。

1D Temporal PE（含 CLS 对齐）已加
使用单独的 CLS 位置编码 + 时序位置编码拼接，避免 CLS 错位。实现位置同上。

Vision Projector 已加
视觉 CLS 全局特征在归一化前经过独立 projector。实现位置同上。

多卡 DDP Gather 对比学习已加
InfoNCE 支持 gather 后扩大负样本池，并处理分布式标签偏移。实现位置同上。

配置已同步
新增并更新了参数（spatial_bev_h/w、gather_ddp）在 configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py。

只保存新增对齐层权重仍有效
保存 Hook 还在，且可用：
projects/mmdet3d_plugin/bevformer/hooks/custom_hooks.py
projects/mmdet3d_plugin/bevformer/hooks/init.py

注册入口已在
projects/mmdet3d_plugin/bevformer/detectors/init.py

1D Temporal PE 的 CLS 对齐约束
已处理为单独的 CLS 位置编码 + 时序位置编码拼接，避免把第 1 帧的位置编码错加到 CLS 上。实现见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。

2D Spatial PE 尺寸匹配约束
已在初始化中创建可学习 Spatial PE（由配置控制，当前是 200x200x256），前向时展平后加到 BEV 稠密特征上，并且加了 HW 一致性检查。实现见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py，配置见 configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py。

Vision Projector 约束
视觉 CLS 全局特征在归一化前经过独立 Vision Projector，已加。实现见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。

多卡 DDP Gather 约束
InfoNCE 已支持分布式 gather，并处理了多卡标签偏移。实现见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。
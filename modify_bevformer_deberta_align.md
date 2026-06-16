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

--------------------------------------------------------------------------------------------------------

date: 202606151520

修改文件：

projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py
本次实现内容：

新增兼容模式参数
在模型里新增模式切换参数：
run_mode
offline_bev_dir
offline_metadata_name
offline_dump_overwrite
支持模式：
origin（别名映射到 online，保持原流程）
offline_extract_bev（别名映射到 extract）
offline_train
offline_infer
offline_infer_validate

新增离线 BEV 特征导出模式
在 extract 模式下，仅提取并保存每个样本的 BEV 特征为 pth。
文件命名为 scene_token_keyframe_nbr.pth。
自动在离线目录维护 bev_feature.json，字段包含：
token（pth 文件 md5）
scene_token
keyframe_nbr
filename
当文件已存在且不覆盖时，会补齐 metadata，避免只存在 pth 但无 json 记录。

新增离线 BEV 特征加载模式
在 offline_train、offline_infer、offline_infer_validate 模式下：
从 bev_feature.json 建立索引并读取对应 pth
支持从 tensor 或常见字典格式载入特征，并校验形状为 T, HW, C。

保持原模式兼容
原训练/推理逻辑在 origin 模式不变。
你文档中“原模式继续保持”的要求已满足。

配置文件已接入可切换参数
默认已设置为 run_mode=origin。
增加了离线特征目录与 metadata 文件名配置位，方便直接切换模式跑流程。

--------------------------------------------------------------------------------------------------------

date: 202606152000

改动位置:
参数扩展在 validate_vlm_align.py:16
新增对齐权重加载函数在 validate_vlm_align.py:37
新增两段加载调度函数在 validate_vlm_align.py:50
主流程改为统一调用新加载逻辑在 validate_vlm_align.py:122

现在的加载行为:
推荐模式：同时给 --base-ckpt 和 --align-ckpt
仅给 --align-ckpt 时：会自动尝试用 位置参数 checkpoint 或 config 的 load_from 作为 base
只给位置参数 checkpoint 时：保持原单 checkpoint 行为
如果没有任何可用 checkpoint：明确报错提示

可信性保障点:
先用 load_checkpoint 加载 base 全量权重
再用 strict=False 叠加 align 权重
会打印 missing_keys 和 unexpected_keys 数量，方便你确认二段覆盖状态是否合理

可信性保障点:
先用 load_checkpoint 加载 base 全量权重
再用 strict=False 叠加 align 权重
会打印 missing_keys 和 unexpected_keys 数量，方便你确认二段覆盖状态是否合理

推荐用法:
两段加载离线验证（你当前需求）
python tools/validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --base-ckpt ./ckpts/epoch_24.pth --align-ckpt work_dirs/xxx/align_trainable_epoch_12.pth
兼容旧用法（单权重）
python tools/validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py work_dirs/xxx/full_or_legacy_checkpoint.pth

改动点:
新增参数 --load-report，在 validate_vlm_align.py:34
新增报告落盘函数，在 validate_vlm_align.py:56
加载流程返回结构化报告（含 align_missing_keys 和 align_unexpected_keys），在 validate_vlm_align.py:65
主流程接入报告输出，在 validate_vlm_align.py:160

你现在可以这样跑:
python tools/validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --base-ckpt ./ckpts/epoch_24.pth --align-ckpt work_dirs/xxx/align_trainable_epoch_12.pth --load-report work_dirs/xxx/val_load_report_epoch12.json

报告里会有:
实际使用的加载模式
base_ckpt 和 align_ckpt 路径
align_missing_keys 详细列表
align_unexpected_keys 详细列表

变更位置:
新增阈值参数在 validate_vlm_align.py:38
新增 unexpected 强校验开关在 validate_vlm_align.py:43
新增加载报告校验函数在 validate_vlm_align.py:74
主流程接入失败检查在 validate_vlm_align.py:199

新增参数:
--max-align-missing-keys N
missing keys 数量大于 N 就退出失败
--fail-on-unexpected-keys
unexpected keys 只要大于 0 就退出失败

推荐命令:
python tools/validate_vlm_align.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --base-ckpt ./ckpts/epoch_24.pth \
  --align-ckpt work_dirs/xxx/align_trainable_epoch_12.pth \
  --load-report work_dirs/xxx/val_load_report_epoch12.json \
  --max-align-missing-keys 0 \
  --fail-on-unexpected-keys

  --------------------------------------------------------------------------------------------------------

date: 202606161315

改动内容:
模型新增参数 text_model_revision 和 text_model_local_files_only
位置：projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
加载 DeBERTa 时把 revision / local_files_only 透传给 from_pretrained
位置：projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
对齐配置暴露这两个开关并加了注释
位置：configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py

如何用:
强可复现（推荐）：
把 configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py 的 text_model_revision=None 改成具体 commit hash。
完全离线：
把 configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py 的 text_model_local_files_only=False 改成 True，并确保本地已有该 revision 的缓存或本地模型目录。

首次运行会把模型文件下载/写入 ckpts/hf_cache。
后续会优先复用该目录缓存。
如果再配合 text_model_local_files_only=True，就可以做到只用本地缓存、不触网（前提是缓存已完整）。

--------------------------------------------------------------------------------------------------------

date: 202606161350

新增文件：
tools/train_validate_vlm_align.py

实现内容：
启动训练（内部固定带 --no-validate，避免走通用 eval hook），入口见 tools/train_validate_vlm_align.py。
轮询 work_dir 下的 align_trainable_epoch_*.pth，见 tools/train_validate_vlm_align.py。
每发现新 checkpoint 就调用 tools/validate_vlm_align.py，见 tools/train_validate_vlm_align.py。
支持把你之前的严格校验参数透传（max missing / unexpected fail），见 tools/train_validate_vlm_align.py。
每轮结果会持续写入 summary JSON（默认 work_dir/align_val_summary.json）。

基础版：
python tools/train_validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --work-dir work_dirs/align_auto_val

严格校验版（推荐）：
python tools/train_validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --work-dir work_dirs/align_auto_val --max-align-missing-keys 0 --fail-on-unexpected-keys --stop-on-val-fail

传额外训练参数（会透传给 train.py）：
python tools/train_validate_vlm_align.py configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py --work-dir work_dirs/align_auto_val -- --cfg-options data.samples_per_gpu=1

说明：
base checkpoint 默认取 config 里的 load_from，也可用 --base-ckpt 显式覆盖。
每个 epoch 的加载诊断报告会写到 work_dir/align_val_load_reports。

--------------------------------------------------------------------------------------------------------

COMMAND:

offline_extract_bev:

mini train split 导出（写入 train 目录）:
python tools/train.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --no-validate \
  --cfg-options \
  model.run_mode=offline_extract_bev \
  model.offline_split=train \
  model.scene_json=data/nuscenes/v1.0-mini/scene.json \
  data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
  data.samples_per_gpu=1 \
  data.workers_per_gpu=0 \
  total_epochs=1 \
  runner.max_epochs=1

mini val split 导出（写入 val 目录）:
python tools/train.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --no-validate \
  --cfg-options \
  model.run_mode=offline_extract_bev \
  model.offline_split=val \
  model.scene_json=data/nuscenes/v1.0-mini/scene.json \
  data.val.ann_file=data/nuscenes/nuscenes_infos_temporal_val.pkl \
  data.samples_per_gpu=1 \
  data.workers_per_gpu=0 \
  total_epochs=1 \
  runner.max_epochs=1

offline validate（默认读 val 目录）:
python tools/validate_vlm_align.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --base-ckpt ./ckpts/epoch_24.pth \
  --align-ckpt work_dirs/xxx/align_trainable_epoch_12.pth \
  --cfg-options model.run_mode=offline_infer_validate

full dataset train split 导出（写入 train 目录）:
python tools/train.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --no-validate \
  --cfg-options \
  model.run_mode=offline_extract_bev \
  model.offline_split=train \
  model.scene_json=data/nuscenes/v1.0-trainval/scene.json \
  data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
  data.samples_per_gpu=1 \
  total_epochs=1 \
  runner.max_epochs=1

full dataset val split 导出（写入 val 目录）:
python tools/train.py \
  configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
  --no-validate \
  --cfg-options \
  model.run_mode=offline_extract_bev \
  model.offline_split=val \
  model.scene_json=data/nuscenes/v1.0-trainval/scene.json \
  data.val.ann_file=data/nuscenes/nuscenes_infos_temporal_val.pkl \
  data.samples_per_gpu=1 \
  total_epochs=1 \
  runner.max_epochs=1

说明：
offline_extract_bev 会被内部映射到 extract，只做 BEV 特征导出。
我给你把 epoch 压到 1，避免重复跑多轮。
导出结果会按 split 写入配置里的 offline_bev_dir_by_split（train/val/test）对应目录。
offline_infer_validate 默认使用 val split 目录。

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

date: 202606181028

把“显存强转 + autocast”改为“权重保持 FP32 + autocast”。
明确写了 AMP 会对不支持 BF16 的算子自动回退到 FP32（包含 nearest2d 场景）。
修正“落盘状态”描述：在该方案下参数保存通常仍是 FP32，而不是“天生纯 BF16”。
实现“autocast 包裹 forward”。
离线 BEV 特征读取也统一到 BF16。

在线推理
权重保持 FP32。前向在 AMP autocast 下运行，主路线会尽量用 BF16。
但为了规避 nearest2d 兼容问题，BEVFormer 提取那一小段被强制为 FP32。

离线提取 BEV
在 use_bf16_amp=True 时，落盘会转成 BF16。
如果是之前已经导出的旧文件，需要覆盖重导出才会变成 BF16。

离线训练、离线推理、离线验证
都走 AMP autocast 路线，离线 BEV 也可直接用 BF16。
但结论是“以 BF16 为主的混合精度”，不是“全链路纯 BF16”，不支持 BF16 的算子仍会回退到 FP32。

--------------------------------------------------------------------------------------------------------

date: 202606181028

修改：

修复离线 train/infer/validate 的读取方式：
现在按 scene_token + frame_nbr 回溯历史帧构建时序片段，不再只读当前帧再硬补到 40。
关键位置：
bevformerv2_debertav3_align.py:609
bevformerv2_debertav3_align.py:612

优化离线 extract 的抽特征路径：
原来每个时间步单独跑一次 backbone；现在每个样本把整段时间帧一次性过 backbone/neck，再按时间步取特征做 BEV。
关键位置：
bevformerv2_debertav3_align.py:713
bevformerv2_debertav3_align.py:715

说明：

我保留了“场景起始处帧数不足时的补齐”机制（仍是 pad 逻辑），但不再是“所有离线样本都只有 1 帧然后复制到 40”。
这次改动已做文件级错误检查，当前无静态报错。

--------------------------------------------------------------------------------------------------------

date: 202606192215

offline trian补未来帧是按真实 meta 索引去补的，不是简单把 1 帧复制成多帧。

当前实现流程是：
先从 bev_feature.json 建 scene -> 可用 frame_nbr 列表
见 bevformerv2_debertav3_align.py:542 和 bevformerv2_debertav3_align.py:544

选窗口时用 past_then_future 策略
先拿历史，不够再补同一 scene 的后续真实 frame_nbr
见 bevformerv2_debertav3_align.py:549 和 bevformerv2_debertav3_align.py:564

对选出的每个 frame_nbr 逐个加载对应 pth
见 bevformerv2_debertav3_align.py:662 和 bevformerv2_debertav3_align.py:675

只有最后兜底这一步，才会在仍不足 expected_frames 时复制最后一帧做 pad（通常发生在该 scene 总帧数本来就不足时）。



anchor 的定义
anchor 不是场景第一帧，也不是场景最后一帧。
anchor 是当前样本里的“当前时刻帧”，也就是时间键 0 对应的那一帧。

为什么是键 0
数据集在组时间队列时先放当前样本到键 0，再补历史帧，然后整体排序。
见 nuscenes_dataset_v2.py:91
见 nuscenes_dataset_v2.py:95
见 nuscenes_dataset_v2.py:110

模型里如何取 anchor
模型优先取键 0 作为 anchor；若异常缺失才退化到最大键。
见 bevformerv2_debertav3_align.py:436
见 bevformerv2_debertav3_align.py:442

在各模式下的作用

online 或 origin
作用：决定本样本的当前时刻，用于组织时序输入和文本对齐目标。
走在线提特征路径。
见 bevformerv2_debertav3_align.py:958
配置默认帧范围是 -39 到 0。
见 bevformerv2-r50-t8-24ep_debertav3_align.py:7

extract 或 offline_extract_bev
作用：用 anchor 定位“当前帧应落盘的那一条离线特征”。
即每个样本按 anchor 导出一帧 pth，并写入对应 frame_nbr 的 metadata。
见 bevformerv2_debertav3_align.py:574
见 bevformerv2_debertav3_align.py:593

offline_train
作用：用 anchor 先映射到 frame_nbr，再围绕该 frame_nbr 组一段离线时序 clip。
现在是历史优先，不足时可补同场景未来真实帧，再最终统一长度。
见 bevformerv2_debertav3_align.py:955
见 bevformerv2_debertav3_align.py:654
见 bevformerv2_debertav3_align.py:659

offline_infer 和 offline_infer_validate
作用和 offline_train 一样，anchor 仍是当前样本时刻，只是用于推理或验证分支。
见 bevformerv2_debertav3_align.py:996
见 bevformerv2_debertav3_align.py:997

一句话总结
anchor 的本质是“这个样本的当前帧基准点”。
extract 用它决定导出哪一帧；offline 模式用它决定从哪一帧向前或向前后组时序。

--------------------------------------------------------------------------------------------------------

COMMAND:

offline_extract_bev:

mini train split 导出（写入 train 目录）:
[ -d data/nuscenes/v1.0-trainval ] || ln -s v1.0-mini data/nuscenes/v1.0-trainval
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--no-validate \
--cfg-options \
model.run_mode=offline_extract_bev \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
model.frames="(0,)" \
data.train.frames="(0,)" \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.samples_per_gpu=1 \
data.workers_per_gpu=8 \
total_epochs=1 \
runner.max_epochs=1

mini val split 导出（写入 val 目录）:
[ -d data/nuscenes/v1.0-trainval ] || ln -s v1.0-mini data/nuscenes/v1.0-trainval
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--no-validate \
--cfg-options \
model.run_mode=offline_extract_bev \
model.offline_split=val \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
model.frames="(0,)" \
data.val.frames="(0,)" \
data.val.ann_file=data/nuscenes/nuscenes_infos_temporal_val.pkl \
data.train.mono_cfg=None \
data.samples_per_gpu=1 \
data.workers_per_gpu=8 \
total_epochs=1 \
runner.max_epochs=1

full dataset train split 导出（写入 train 目录）:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--no-validate \
--cfg-options \
model.run_mode=offline_extract_bev \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-trainval/scene.json \
model.frames="(0,)" \
data.train.frames="(0,)" \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.samples_per_gpu=1 \
data.workers_per_gpu=8 \
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
model.frames="(0,)" \
data.val.frames="(0,)" \
data.val.ann_file=data/nuscenes/nuscenes_infos_temporal_val.pkl \
data.val.mono_cfg=None \
data.samples_per_gpu=1 \
data.workers_per_gpu=8 \
total_epochs=1 \
runner.max_epochs=1

offline train:

mini:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--no-validate \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True \
data.samples_per_gpu=2 \
data.workers_per_gpu=2 \
data.persistent_workers=False \
data.prefetch_factor=1


说明：
offline_extract_bev 会被内部映射到 extract，只做 BEV 特征导出。
我给你把 epoch 压到 1，避免重复跑多轮。
导出结果会按 split 写入配置里的 offline_bev_dir_by_split（train/val/test）对应目录。
offline_infer_validate 默认使用 val split 目录。

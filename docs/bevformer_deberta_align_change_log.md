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

date: 202606202115

数据集新增无重叠采样开关
在 nuscenes_dataset_v2.py:16 增加参数：
offline_unique_anchor
offline_drop_last_chunk
在 nuscenes_dataset_v2.py:41 新增 _build_offline_chunks：
按 scene 分组
按 len(frames) 分块（例如 40 帧一块）
每块只对应 1 个训练样本（非滑窗）
在 nuscenes_dataset_v2.py:60 改写 len：
开启 offline_unique_anchor 且 offline_meta_only 时，长度变为 chunk 数
在 nuscenes_dataset_v2.py:67 改造 _prepare_offline_meta_only：
构造块内固定时序元数据
把精确 frame 列表写入 anchor meta 的 offline_frame_nbrs
尾块可选：
drop（offline_drop_last_chunk=True）
pad 最后一帧（False）
检测器支持按显式 frame 列表读取（避免再次滑窗重叠）
在 bevformerv2_debertav3_align.py:656 的 _load_offline_bev_sequence 中：
若 meta 含 offline_frame_nbrs，优先使用该列表加载
否则保持旧逻辑 _select_offline_frame_numbers
配置暴露开关
在 bevformerv2-r50-t8-24ep_debertav3_align.py:67 的 train 数据配置新增：
offline_unique_anchor=False
offline_drop_last_chunk=False

当同时开启这两个条件时：

data.train.offline_meta_only=True
data.train.offline_unique_anchor=True
数据集不再按滑窗随机取任意 anchor，而是先按 scene 切成不重叠的 40 帧块，每个块只生成 1 个样本（1 个“anchor 样本”），实现位置在 nuscenes_dataset_v2.py:41 和 nuscenes_dataset_v2.py:67。
模型侧会优先按这个样本里显式给的 frame 列表读取，不再二次滑窗，见 bevformerv2_debertav3_align.py:656。

所以你说的这句在该模式下成立：

这个 epoch 内，已经训练过的那一块，不会再被“随机 anchor”重复补齐训练。
两个边界注意点：

如果尾块不足 40 且 offline_drop_last_chunk=False，会在该尾块内部用最后一帧做 pad（只影响尾块）。
多卡 DDP 下，若 sampler 需要补齐到可整除，极少量块可能被重复分配一次（这是采样器层面的）。单卡时最干净。

默认模式（不打开 offline_unique_anchor）
是，逻辑保持不变。还是按原来的 anchor 选帧策略补齐，走 detector 里的选择逻辑（past_only 或 past_then_future），入口在 bevformerv2_debertav3_align.py:656。

新的无重叠模式（offline_meta_only=True 且 offline_unique_anchor=True）
这里不再用“anchor 前后补齐的滑窗逻辑”来构造训练片段，而是 dataset 先按 40 帧分块并把明确帧号传下去，位置在 nuscenes_dataset_v2.py:41 和 nuscenes_dataset_v2.py:67。
detector 会优先用这份显式帧号，不再二次按 anchor 滑窗补齐，判断点在 bevformerv2_debertav3_align.py:661。

一句话总结：默认不变；开启无重叠开关后，补齐逻辑改为“按块固定帧列表”，不是原先的 anchor 滑窗补齐。



按“开启无重叠开关后，一条样本从生成到读取”的完整链路详细讲。

前提条件（同时满足）：

data.train.offline_meta_only=True
data.train.offline_unique_anchor=True
配置入口在 bevformerv2-r50-t8-24ep_debertav3_align.py:67。

1. 先按 scene 建“不重叠分块”
实现在 nuscenes_dataset_v2.py:41。

逻辑是：

把全量 data_infos 按 scene_token 分组。
每个 scene 内按 frame_idx 升序排列。
以 queue_length（你的配置里是 40）为步长切块。
也就是第 1 块 [0:40]，第 2 块 [40:80]，第 3 块 [80:120]……
每个块只产生 1 个训练样本，不再滑窗偏移 1 帧去生成下一条样本。
这一步已经从根上消除了“anchor 滑动导致同一帧跨样本重复出现”的主来源。

2. 数据集长度改为“块数”
实现在 nuscenes_dataset_v2.py:60。

也就是说一个 epoch 迭代的不是原始帧数，而是 chunk 数。
每个 chunk 对应一次训练输入。

3. 每个 chunk 显式产出固定帧列表
实现在 nuscenes_dataset_v2.py:67。

关键点：

取当前 chunk 的 index 列表。
按 frames 的时序键（-39 到 0）把这些帧填进 temporal meta。
同时把真实 frame_nbr 列表写到 anchor meta 里：offline_frame_nbrs（见 nuscenes_dataset_v2.py:95）。
这相当于告诉后端模型：“这条样本就用这 40 个明确帧号，不要你再推导。”

4. detector 优先用显式帧列表，不再按 anchor 二次选帧
实现在 bevformerv2_debertav3_align.py:656。

行为是：

如果检测到 offline_frame_nbrs，就直接按这个列表加载 pth。
只有没给 offline_frame_nbrs 才回退到旧逻辑 _select_offline_frame_numbers（past_only / past_then_future）。
所以开启无重叠后，旧的“anchor 前后补齐策略”在训练主路径里基本被旁路了。

5. 尾块处理（很关键）
由 offline_drop_last_chunk 控制，配置在 bevformerv2-r50-t8-24ep_debertav3_align.py:72。

False：尾块不足 40 时，用尾块最后一帧补齐到 40（不丢数据，但尾块内部会有重复）。
True：直接丢弃不足 40 的尾块（最严格“每真实帧最多一次”，但会损失少量样本）。

6. 你关心的“每帧只训练一次”到底如何成立

对大部分完整块：每个真实帧只属于一个 chunk，因此每 epoch 只出现一次。
可能破坏严格性的只有两处：
尾块 pad（当 offline_drop_last_chunk=False）。
DDP 分布式采样器为凑整可能带来极少重复（采样器层面，不是 chunk 逻辑本身）。
如果你要最严格版本，建议：

offline_drop_last_chunk=True
单卡或确认分布式 sampler 不做重复补齐（或可接受极小重复）



这一轮你提的“补 val loss + 全局 recall”改动里，没有改训练计算逻辑。

具体是：

改了验证/评估链路
tools/validate_vlm_align.py
用途：从 batch 内指标改为可做全局检索矩阵评估，并打印 val_loss_align / R@K。
改了测试输出字段（给验证脚本用）
projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
用途：在 forward_test 里额外返回 vision/text embedding 和 loss_align，用于 val 端聚合。
注意：forward_train 没动。
改了联训联验脚本的汇总解析
tools/train_validate_vlm_align.py
用途：解析并记录 val_loss_align、切换验证模式。
注意：不改变训练 loss 计算。
所以训练侧的 loss 定义、反向传播、优化器更新逻辑都没改。


--------------------------------------------------------------------------------------------------------

date: 202606202115

先说结论

你当前实现已经有跨卡 gather 和对称 InfoNCE，基础很好。入口在 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。
队列改造建议做成 可开关、短队列、双向队列、跨卡入队、同场景负样本掩码（可选） 这 5 个点。
训练建议仍保持每卡 batch 一致，队列只负责放大负样本池，不替代 DDP 基本对称性。
落地步骤

在模型初始化里加队列参数与缓冲区

文件: projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

在现有 logit_scale 附近新增：
use_feature_queue: bool
queue_size: int（建议先 128）
queue_warmup_steps: int（建议先 50）
queue_use_scene_mask: bool（建议先 True）

register_buffer 新增：
text_queue, vision_queue，形状 [K, C]
text_queue_scene_id, vision_queue_scene_id，形状 [K]
queue_ptr, queue_valid_len, global_step

新增队列更新函数（必须处理回绕）:

文件同上
新增 _dequeue_and_enqueue(keys, scene_ids, queue_name)
要点：
keys 必须先 detach 再入队
批量写入时处理 ptr + batch_size 超过 K 的回绕
更新 queue_valid_len，避免 warmup 初期读到未填满区域

改造损失函数为 当前批 + 队列负样本
当前函数在 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
目标逻辑：
先保留你已有的 gather_ddp，拿到 global_vision/global_text
i2t: logits = [vision 对 global_text] 拼接 [vision 对 text_queue_valid]
t2i: logits = [text 对 global_vision] 拼接 [text 对 vision_queue_valid]
labels 仍指向 global 部分的正样本，不指向队列区
关键细节：
队列部分只作为负样本，必须无梯度
如果启用 scene mask，把 与当前样本同 scene 的队列项 置为极小值，避免假负样本污染

在 forward_train 里补 scene_id 并入队
训练入口在 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py
做法：
复用你现有 scene_token 解析逻辑，拿到每个样本的 scene_token
映射成稳定 scene_id（可用哈希后取 int64）
loss 计算完成后，用 gathered 的当前批 text/vision 特征和 scene_id 入队
先更新 global_step，再按 warmup 决定是否启用队列 logits（前几步仅用原始 global batch）


配置层新增开关
配置文件: configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py
新增建议默认值：
use_feature_queue=True
queue_size=128
queue_warmup_steps=50
queue_use_scene_mask=True

验证与推理保持不变
forward_test 里继续不用队列，保持你现在的全局评估语义，避免指标定义混淆。位置在 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py。

参数建议（首轮）

queue_size: 128（12GB 显存更稳）
accumulation_steps: 2 或 4（和队列可并用）
每卡 batch: 1（异构卡下最稳）
温度仍用你当前 learnable logit_scale，不必改

你这个架构下的注意点

冻结 backbone 只说明骨干特征稳定，不代表 projector 后特征完全不漂移。
所以队列不要太长，先 128；太长会加重陈旧特征副作用。
scene 去重掩码很重要，否则同场景文本容易被当负样本。

验收标准（建议一次性检查）

打印每步有效负样本数，确认从 global_batch 扩展到 global_batch + queue_valid_len
监控 loss_align 曲线是否更平滑
监控 i2t_top1 和 t2i_top1 是否比无队列版本稳定提升
确认 DDP 下无 shape mismatch、无死锁



已完成改动

模型新增队列参数（可开关）
在构造函数中加入：
use_feature_queue
feature_queue_size
feature_queue_warmup_steps
位置: bevformerv2_debertav3_align.py:45

新增 FIFO 队列缓冲区与更新逻辑
新增 buffer：
text_queue
vision_queue
queue_ptr
queue_valid_len
queue_step
新增函数：
_gather_no_grad
_dequeue_and_enqueue
_enqueue_feature_queue
_queue_ready
位置: bevformerv2_debertav3_align.py:268, bevformerv2_debertav3_align.py:924

对比损失支持“当前批 + 队列负样本”
_contrastive_loss 新增 use_queue 开关
训练时可把队列负样本拼到 logits 右侧
labels 仍指向当前 global batch 正样本区域
位置: bevformerv2_debertav3_align.py:985

训练启用队列，测试关闭队列
forward_train:
use_queue=self.use_feature_queue
每步 loss 后入队并 step+1
位置: bevformerv2_debertav3_align.py:1071
forward_test:
明确 use_queue=False，保持验证语义稳定
位置: bevformerv2_debertav3_align.py:1121

配置已暴露并默认开启最小队列
use_feature_queue=True
feature_queue_size=128
feature_queue_warmup_steps=50
位置: bevformerv2-r50-t8-24ep_debertav3_align.py:47


把 scene mask 版本落地了，当前是“队列 + scene mask”可开关实现，默认在配置里已开启。

本次改动

新增模型参数 queue_use_scene_mask（默认 False）
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

队列新增 scene id 缓冲区
新增 text_queue_scene_id / vision_queue_scene_id
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

新增 scene token 到稳定 int64 id 的映射与解析
新增 _resolve_scene_tokens / _scene_token_to_id / _resolve_scene_ids
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

队列入队改为同时写入 feature 和 scene id
改造 _dequeue_and_enqueue / _enqueue_feature_queue
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

对比损失接入 scene mask
在队列负样本 logits 上对同 scene 项做 masked_fill(-1e4)
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

训练前向接入 scene_ids
forward_train 已传入 scene_ids 到 loss 和 enqueue
见 projects/mmdet3d_plugin/bevformer/detectors/bevformerv2_debertav3_align.py

配置已开启 scene mask
queue_use_scene_mask=True
见 configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py

--------------------------------------------------------------------------------------------------------

date: 202606232100

现在改过的脚本逻辑：

串行 val 只在 rank0（NODE_RANK=0）执行。
画图/报表导出也只在 rank0 执行。
因此这些结果文件只会写到 rank0 的 WORK_DIR。
rank1 的行为是：

只参与训练阶段 DDP
训练后跳过 validate/export，等待下一轮训练 rendezvous
所以你看到的 val_logs、val_metrics_json、val_metrics.tsv、train_val_compare.tsv/png/md 都在 rank0 那边。
如果需要 rank1 也有一份，就训练后再从 rank0 同步这些产物过去即可。

只同步多机共同训练的必要文件
每个 epoch 仅同步 epoch_${epoch}.pth（这是下一轮 resume 必需文件）。
不再默认同步 align_trainable_epoch_*.pth / latest.pth。
使用 rsync password-file（daemon 模式）支持多机串行多轮 epoch
已在脚本内加入自动同步逻辑（仅 rank0 执行）。
非 rank0 节点在下一轮 resume 前会等待断点文件到位，避免“文件未同步就报错”。

新增可配置参数（含默认值）：

ENABLE_RSYNC_SYNC=false
RSYNC_PASSWORD_FILE=/etc/rsync.password
RSYNC_TARGET_IP=192.168.103.3
RSYNC_TARGET_USER=rsync_user
RSYNC_TARGET_PATH=backup_module/
RSYNC_EXTRA_OPTS=-avz
RESUME_WAIT_SECONDS=3600
RESUME_WAIT_INTERVAL=10

epoch_N.pth 是续训硬依赖
下一轮训练 --resume-from 用的是这个文件。
所以它必须同步，这也是现在只同步它的原因。
位置见 train_val_epoch_serial_ddp.sh:188
align_trainable_epoch_N.pth 主要用于验证/导出
你现在多机模式下只在 rank0 做 val 和报表。
rank1 不做 val，因此 rank1 不需要这份文件也能参与下一轮训练。
位置见 train_val_epoch_serial_ddp.sh:221
latest.pth 不是当前续训路径必需
你的脚本续训依赖的是明确的 epoch_${prev}.pth，不是 latest.pth。
所以不同步 latest.pth 不会卡住当前串行多轮 epoch 训练。



按“一个 epoch 结束后”把两个 rank 的时序拆开讲。

当前脚本里的关键逻辑点：

rank0 推送：sync_required_ckpt_to_peer "${WORK_DIR}/epoch_${epoch}.pth"
rank1 等待：下一轮开始前 wait_for_resume_ckpt "${WORK_DIR}/epoch_${prev_epoch}.pth"
rank1 跳过 val：if [[ NNODES>1 && NODE_RANK!=0 ]]; then continue

执行时序（以 epoch=2 结束为例）

两边先完成 epoch=2 的 DDP 训练
torchrun 训练阶段是同步的，直到两边都退出本轮 train 命令。

rank0 训练返回后先做 rsync
rank0 调用 sync_required_ckpt_to_peer，只传 epoch_2.pth。
这个 rsync 命令是前台阻塞执行：没传完不会返回。
所以 rank0 不会立刻去做 val 或下一轮。

rank1 训练返回后不做 val
rank1 命中“多机非0号节点跳过验证”的分支，直接 continue 到下一轮循环。
它不会在本轮做任何 rsync 推送，也不会写 val 报表。

到 epoch=3 开始时，rank1 先检查 resume 文件
因为 epoch>1，rank1 会先执行 wait_for_resume_ckpt WORK_DIR/epoch_2.pth（仅在 ENABLE_RSYNC_SYNC=true 时）。
如果文件已到位，立即继续；如果还没到，会按 RESUME_WAIT_INTERVAL 轮询，最多等 RESUME_WAIT_SECONDS。
超时就报错退出，防止拿不到断点却硬跑。

rank0 在 rsync 完成后才继续
rank0 接着做本轮 val（只 rank0 做）。
val 完后进入下一轮，读取同一个 epoch_2.pth 做 --resume-from。
此时 rank1 通常也已经等到该文件，两个节点能在下一次 torchrun rendezvous 汇合。

为什么这样设计是安全的:

续训必需文件只有 epoch_N.pth，只同步它就够下一轮 train。
rsync 阻塞保证 rank0 不会“还没传完就进入后续逻辑”。
rank1 有显式等待，不会因为文件慢到而直接失败。
val 只在 rank0，避免多节点重复验证、重复写报表冲突。

边界与注意点:

这个机制默认是“rank0 -> 一个目标节点（你配置的 RSYNC_TARGET_IP）”。
如果未来是 3+ 节点，需要扩展为推送到多个目标，或改共享存储。
如果 rank1 的 WORK_DIR 与 rsync 落地路径不一致，等待会一直看不到文件。
如果不用 rsync（ENABLE_RSYNC_SYNC=false），则要你自己保证 rank1 上 epoch_N.pth 已存在。

--------------------------------------------------------------------------------------------------------

COMMAND:

- offline_extract_bev:

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

说明：
offline_extract_bev 会被内部映射到 extract，只做 BEV 特征导出。
我给你把 epoch 压到 1，避免重复跑多轮。
导出结果会按 split 写入配置里的 offline_bev_dir_by_split（train/val/test）对应目录。
offline_infer_validate 默认使用 val split 目录。

- offline train:

DDP 2x3080Ti + 1x3090，v1.0-mini，offline train：

mini:
第一次训练（生成可续训的全量 checkpoint）
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
torchrun --nproc_per_node=3 --master_port=29501 \
tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--launcher pytorch \
--work-dir work_dirs/mini_ddp_offline_train \
--no-validate \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True \
total_epochs=1 \
runner.max_epochs=1

下次训练（加载上次全量 checkpoint 继续）：
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
torchrun --nproc_per_node=3 --master_port=29501 \
tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--launcher pytorch \
--work-dir work_dirs/mini_ddp_offline_train \
--resume-from work_dirs/mini_ddp_offline_train/epoch_1.pth \
--no-validate \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True \
total_epochs=2 \
runner.max_epochs=2



- offline train and val:

mini:
第一次：
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train_validate_vlm_align.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--work-dir work_dirs/mini_auto_val_fullckpt \
--base-ckpt ./ckpts/bevformer/epoch_24.pth \
--validate-after-train \
-- \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True

后续:
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
python tools/train_validate_vlm_align.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--work-dir work_dirs/mini_auto_val_fullckpt \
--base-ckpt ./ckpts/bevformer/epoch_24.pth \
--validate-after-train \
-- \
--resume-from work_dirs/mini_auto_val_fullckpt/epoch_1.pth \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True \
total_epochs=2 \
runner.max_epochs=2

ddp 3卡：
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 \
torchrun --nproc_per_node=3 --master_port=29501 \
tools/train.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--launcher pytorch \
--work-dir work_dirs/mini_ddp_offline_train_val \
--no-validate \
--cfg-options \
model.run_mode=offline_train \
model.offline_split=train \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.train.ann_file=data/nuscenes/nuscenes_infos_temporal_train.pkl \
data.train.mono_cfg=None \
data.train.offline_meta_only=True \
total_epochs=1 \
runner.max_epochs=1 \
&& \
python tools/validate_vlm_align.py \
configs/bevformer_vlm_align/bevformerv2-r50-t8-24ep_debertav3_align.py \
--base-ckpt ./ckpts/bevformer/epoch_24.pth \
--align-ckpt work_dirs/mini_ddp_offline_train_val/align_trainable_epoch_1.pth \
--cfg-options \
model.run_mode=offline_infer_validate \
model.offline_split=val \
model.scene_json=data/nuscenes/v1.0-mini/scene.json \
data.val.offline_meta_only=True


每个 epoch 串行执行一次 train，然后立即执行一次 val，再进入下一轮：

mini：
START_EPOCH=1 \
END_EPOCH=3 \
DATASET_PROFILE=mini \
CUDA_VISIBLE_DEVICES=0,1,2 \
BASE_CKPT=./ckpts/bevformer/epoch_24.pth \
./tools/train_val_epoch_serial_ddp.sh

如果继续是：
START_EPOCH=4 \
END_EPOCH=6 \
DATASET_PROFILE=mini \
CUDA_VISIBLE_DEVICES=0,1,2 \
BASE_CKPT=./ckpts/bevformer/epoch_24.pth \
./tools/train_val_epoch_serial_ddp.sh

trainval：
START_EPOCH=1 \
END_EPOCH=1 \
DATASET_PROFILE=trainval \
CUDA_VISIBLE_DEVICES=0,1,2 \
BASE_CKPT=./ckpts/bevformer/epoch_24.pth \
./tools/train_val_epoch_serial_ddp.sh

多PC，需要PC之间保持代码版本和路径、数据集路径、中间结果（work_dirs）一致：
主节点：
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.103.2 \
MASTER_PORT=29501 \
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
ENABLE_RSYNC_SYNC=true \
RSYNC_PASSWORD_FILE=/etc/rsync.password \
RSYNC_TARGET_IP=192.168.103.3 \
RSYNC_TARGET_USER=rsync_user \
RSYNC_TARGET_PATH=backup_module/ \
START_EPOCH=1 \
END_EPOCH=3 \
DATASET_PROFILE=trainval \
BASE_CKPT=./ckpts/bevformer/epoch_24.pth \
./tools/train_val_epoch_serial_ddp.sh

从节点:
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.103.2 \
MASTER_PORT=29501 \
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
ENABLE_RSYNC_SYNC=true \
START_EPOCH=1 \
END_EPOCH=3 \
DATASET_PROFILE=trainval \
BASE_CKPT=./ckpts/bevformer/epoch_24.pth \
./tools/train_val_epoch_serial_ddp.sh


可直接用这条命令就行（可写到任意绝对路径，避免 work_dir 权限问题）：
python tools/export_train_val_compare.py \
--work-dir work_dirs/mini_ddp_offline_train_val \
--output train_val_compare_mini.tsv

说明：
这个表按 epoch 合并了 train 与 val 指标。
train 侧给了末次与均值字段（如 train_last_loss_align、train_avg_loss_align）。
val 侧给了 val_loss_align、R@1/R@5/R@10 等。
你之前旧轮次没有 train 指标时会显示 NA，这是正常的，因为当时日志没有被该脚本统计到该轮可用行。
你现在新跑的轮次会持续可对齐，便于直接和 train 做横向对比。

生成图：可直接用这条命令重复生成：
python tools/plot_train_val_compare.py \
--compare-file train_val_compare_mini.tsv \
--output train_val_compare_mini.png

- 读表：
epoch
表示训练轮次编号。

train_last_loss_align
该 epoch 内“最后一次被日志打印到”的对齐损失（train 侧）。
更像“该轮末尾状态”。

train_avg_loss_align
该 epoch 内所有日志点的 loss_align 平均值（train 侧）。
更像“该轮整体水平”。

val_loss_align
验证集上的对齐损失（val 侧，全局评估结果）。

train_last_acc_i2t_top1
训练侧图到文 Top-1 准确率（i2t），取该轮最后一次日志值。
含义：图像特征检索文本时，Top-1 命中的比例。

val_i2t_top1
验证侧图到文 Top-1 准确率（i2t）。

train_last_acc_t2i_top1
训练侧文到图 Top-1 准确率（t2i），取该轮最后一次日志值。
含义：文本特征检索图像时，Top-1 命中的比例。

val_t2i_top1
验证侧文到图 Top-1 准确率（t2i）。

快速读表建议：

先看 val_loss_align 和 val_i2t_top1/val_t2i_top1 判断泛化是否变好。
再看 train_last_loss_align vs train_avg_loss_align 判断该轮末尾是否偏离该轮平均。
若 train 明显变化但 val 几乎不变，通常是“训练拟合在动、泛化没提升”。

判断过拟合时，建议主看这组关系：

train_avg_loss_align
看训练集整体是否持续下降（比 train_last_loss_align 更稳）。

val_loss_align
看验证集是否同步下降。
如果 train 降、val 不降甚至升，才是典型过拟合信号。





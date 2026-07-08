---
name: BEVFormer-v2-DeBERTa-v3-Scenarios-Texts-Two-Towers-Match-Training-Validation
description: 利用 BEVFromer-v2 和 DeBERTa-v3 做时序场景文字匹配模型训练
tools: [ "#codebase", "#terminal" ]
---

# Role
你的任务是：
- 基于当前BEVFormerv2进行修改，编写 Python，Pytorch 程序。
- 程序的作用之一是训练一个模型；这个模型的作用是对于6个camera，2Hz Keyframe，20秒的场景数据，做文字匹配。
- 程序的作用之一是验证训练后的模型，做Validation。

# 训练方案架构描述

## 当前实现更新（2026-07-08）

当前对齐模型已经从早期的“Temporal CLS + Vision/Text Projector”单一路径，扩展为可配置的多种视觉池化和诊断友好结构。默认实验配置重点用于缓解 vision embedding collapse / hubness：

- 视觉侧支持 `vision_pooling={cls, frame_mean, spatial_mean, temporal_attn, netvlad}`。
- 当前推荐配置使用 `vision_pooling=netvlad`，通过多簇残差池化聚合时序帧 token。
- `use_vision_projector` 和 `use_text_projector` 都是可开关项；当前 NetVLAD 结构实验中默认关闭 projector，减少额外投影层把特征压到公共方向的风险。
- 新增 `vision_batch_centering_weight` 和 `vision_covariance_reg_weight`，用于约束视觉 embedding 的公共方向、低方差维度和维度相关性。
- 验证脚本可导出最终 embedding 与中间层 embedding 诊断，用于定位 collapse 发生在 spatial pooling、temporal encoder、NetVLAD、projector 还是 normalize 后。

## 流程图1

【 图像塔 (Vision Tower) 】                               【 文本塔 (Text Tower) 】

           Cam 环视图像 (40帧)                                      场景文本真值描述
                │                                                    (来自 scene.json)
                ▼                                                           │
          BEVFormer-v2                                                      ▼
     [40, 40000, 256] 稠密特征                                       DeBERTa-v3-base
                │                                                           │
                ├─◄─── 注入 2D 空间位置编码 (Spatial PE)                     │
                ▼                                                           ▼
       空间 Attention Pooling 压扁                                   [B, Seq_Len, 768]
                │                                                           │
                ▼                                                           ▼ Masked Mean Pooling
         [40, 256] 时序骨架                                           (过滤 [PAD] 占位符)
                │                                                           │
                ▼ Linear 线性层直接升维                                      ▼
            [40, 768]                                                [B, 768] 变长句向量
                │                                                           │
                ▼ 拼入 [CLS] Token                                         ▼
            [41, 768]                                                 Text Projector
                │                                                 (独立对齐映射层, 参与训练)
                ├─◄─── 注入 1D 时序位置编码 (Temporal PE)                   │
                ▼                                                           ▼
     3层 Lightweight Transformer                                       L2 Normalize
    [41, 768] 帧间/全局多层交互                                              │
                │                                                           ▼
                ▼ 多策略视觉池化                                      [B, 768] 最终文本向量
            [B, 768]                                                        │
                │                                                           │
                ▼ Optional Vision Projector                                 │
            (可开关/可残差插值的对齐映射层)                                    │
                │                                                           │
                ▼                                                           │
           L2 Normalize                                                     │
                │                                                           │
                ▼                                                           │
       [B, 768] 最终视觉向量                                                 │
                │                                                           │
                └───────────────────────► 【 矩阵点积 】 ◄──────────────────┘
                                        (计算余弦相似度)
                                                │
                                                ▼
                                        InfoNCE Loss 对比损失
                                  (支持多卡 DDP Gather 扩大 Batch)

## 图像塔数据流
图像塔的目标是将长时序的自动驾驶环视画面，压缩、精炼并升维成一个能够代表整个视频片段全局语义的 [B, 768] 向量。

1. 多视角时序输入
- 系统输入为连续 20 秒、采样率为 2Hz 的 6 路摄像头环视图像，总计包含 40 帧 视频画面（Cam 环视图像）。

2. 稠密 BEV 特征提取
- 图像输入送入 BEVFormer v2（作为视觉骨干网络）。它将多视角图片信息投影到统一的 3D 鸟瞰图（BEV）空间。为了在 3090 训练时节省显存，特征保持在较低的 256 维。输出的稠密特征形状为 [40, 40000, 256]，代表 40 个时间帧，每帧有 40,000 个 BEV 网格（如 200 \times 200），每个网格为 256 维。
- 需要注意的是，在当前配置下，BEVFormer v2被完全冻结权重，仅进行前向传播以提取特征。

3. 空间 Attention Pooling 压扁
- 在空间池化前注入2D 空间位置编码 (Spatial PE)，确保模型压缩了空间维度后，依然能识别出文本中“左前”、“右后”等关键方位。
- 为了消除密集的空间维度，引入一个可学习的全局 Query 向量，通过空间自注意力机制（Spatial Attention Pooling），把每帧的 40,000 个点融合、压缩为 1 个点。此时，张量形状“压扁”为 [40, 256]，被称为时序骨架。

4. 线性层直接升维
- 通过一个全连接层（Linear 线性层），将特征从 256 维直接拉伸升维到 768 维，得到形状为 [40, 768] 的张量。此时，空间维度已完全消失，仅保留 40 个纯粹的时间序列 Token。

5. 拼入 [CLS] Token 与位置编码
- 在 40 个时序 Token 的最前端（第 0 位）拼接一个全零初始化的可学习 [CLS]（分类标记） 向量，序列长度由 40 变成 41，形状变为 [41, 768]。紧接着，为这 41 个位置注入绝对或相对时序位置编码。

6. 时序 Transformer 会议交互
- 在时序 Transformer 前注入1D 时序位置编码 (Temporal PE)，确保模型能识别“先做什么、后做什么”的动作先后顺序。
- 将 [41, 768] 的张量送入一个由 3 层组成的轻量级时序 Transformer Encoder。在多层 Self-Attention 的作用下，第 0 位的 [CLS] Token 吸取和概括后面 40 帧图像中的所有因果、时序、空间变化信息。

7. 多策略视觉池化并归一化
- 当前实现不再只固定提取第 0 位 [CLS] 向量，而是通过 `vision_pooling` 选择视觉全局表征：
    - `cls`: 使用 temporal encoder 输出的第 0 位 [CLS] token。
    - `frame_mean`: 对 temporal encoder 输出的帧 token 求均值。
    - `spatial_mean`: 在时序编码前，对 spatial pooling 后的帧表征求均值。
    - `temporal_attn`: 使用一个可学习 temporal query 对帧 token 做 MultiheadAttention 池化。
    - `netvlad`: 对 temporal encoder 输出的帧 token 做多簇残差池化，再投影回 768 维。
- 当前抗 collapse 实验推荐使用 `netvlad`，其目标是避免所有片段被压缩到单一全局方向。

8. Optional 投影映射层与归一化
- Vision Projector 现在是可选模块。`use_vision_projector=True` 时，可通过 `vision_projector_residual_weight` 在 raw vision feature 和 projector output 之间做残差插值；`False` 时直接使用池化后的 raw vision feature。
- 最终统一经过 L2 Normalize（L2 归一化），消除尺度影响，得到用于对齐的 [B, 768] 最终视觉向量。

## 文本塔数据流
文本塔的目标是将自然语言描述，转化为与视觉向量处于同一几何超平面的 [B, 768] 向量。

1. 真值描述输入
- 从nuscenes数据集的 scene.json 元数据中提取出对应场景的文本真值描述（如“主车直行，遭遇暴雨，右侧有重型卡车”）。

2. 预训练文本大模型编码
- 将文本送入 DeBERTa-v3-base 骨干网络。
- 需要注意的是，在当前配置下，DeBERTa-v3被完全冻结权重，仅进行前向传播以提取特征。

3. Token 特征输出
- DeBERTa 输出分词后的 Token 级特征，张量形状为 [B, Seq_Len, 768]。其中 Seq_Len 是当前 Batch 中最长句子的长度（短句子会在后方填充 [PAD] 占位符）。

4. Masked Mean Pooling 聚合
- 为了得到整句话的宏观表征，系统通过 Masked Mean Pooling（掩码平均池化） 顺着序列维度求均值。该机制会利用 attention_mask 将所有的 [PAD] 占位符特征乘以 0 抹杀，只对真正有物理意义的字词特征求中心质心。

5. 生成固定句向量
- 消除变长序列长度后，张量完美坍缩为固定大小的 [B, 768] 变长句向量。

6. Optional 投影映射层与归一化
- Text Projector 现在是可选模块。`use_text_projector=True` 时使用可训练 Text Projector；`False` 时直接使用 DeBERTa masked mean pooled feature。
- 随后同样进行 L2 Normalize（L2 归一化），最终吐出标准规格的 [B, 768] 最终文本向量。

## 视觉几何正则

为缓解训练中观察到的 vision embedding 近似同向、检索结果 hubness 严重等问题，当前实现新增两个视觉侧辅助 loss：

- `loss_vision_centering`: 惩罚 batch 内视觉特征均值向量的平方范数，降低 dominant common direction。
- `loss_vision_covariance`: 对视觉特征先按维度标准化，再惩罚低方差维度和 off-diagonal correlation，形式接近 VICReg 的 variance/covariance 约束。

推荐起始配置：

- `vision_batch_centering_weight=0.05`
- `vision_covariance_reg_weight=0.04`

注意事项：

- 几何正则默认使用 gather 后的全局 batch 特征，与 `gather_ddp=True` 配合。
- 该正则作用于 vision feature geometry，不改变验证阶段的相似度计算方式。
- 若 loss_align 明显不收敛，需要分别关掉 centering/covariance 做消融，判断是否正则权重过强。

## 跨模态对齐交互
当左塔吐出 [B, 768] 的视觉向量，右塔吐出 [B, 768] 的文本向量后，两边在最底层完成闭环。

1. 矩阵点积（余弦相似度）
- 将图像塔矩阵与转置后的文本塔矩阵进行矩阵乘法（点积）。因为两边在前面都做过了 L2 归一化，此时相乘矩阵内的每一个元素代表的正是“第 i 个视频片段与第 j 句文本之间的余弦相似度（Cosine Similarity）”，最终形成一个 [B, B] 的全局相似度对齐得分矩阵。

2. 对比损失计算
- 在这个 [B, B] 的对齐矩阵上，正对角线（i=j 的位置）代表图像和文本完美匹配的真值，其余位置均为负样本。模型通过计算 InfoNCE Loss（对比损失函数），以极高的梯度拉近正样本对在 768 维空间中的距离，同时推开负样本对。

3. 支持多卡 DDP Gather

## 权重文件
从物理文件上看，你的项目涉及 3 个部分的权重。

1. BEVFormer v2 的权重
- 状态：完全没有变动。因为在整个训练过程中把它冻结了（requires_grad=False），它的权重参数从第一步到最后一步没有发生任何修改。
- 处理：训练完后不需要保存它。（因为手头本来就有官方或你之前练好的预训练 .pth 或 .bin 文件）。
- 路径：ckpts/epoch_24.pth。

2. DeBERTa-v3 的权重
- 状态：完全没有变动。同理，它也被全程冻结，参数和 HuggingFace 官方下载下来的时候一模一样。
- 处理：训练完后完全不需要保存。
- 路径：通过 Hugging Face 下载 “microsoft/deberta-v3-base” 相关。

3. 中间新增的对齐环节
- 包含内容：空间 Attention Pooling 层、Linear 升维层（256 -> 768）、3层轻量级时序 Transformer、Temporal Attention Pooling、NetVLAD Pooling、Vision/Text Projector、logit_scale、Feature Queue 等对齐相关参数。
- 状态：大幅度更新。这是整个训练过程中唯一在接收梯度、不断优化的部分。
- 处理：唯一需要使用 torch.save() 保存的就是这一份权重。

当前 `trainable_state_dict()` 需要覆盖新增的 `temporal_pool_query`、`temporal_pool`、`vision_netvlad_assign`、`vision_netvlad_clusters_param`、`vision_netvlad_proj`，避免只保存旧版本对齐头导致恢复训练缺参数。

## 配置文件
1. 1份配置文件
- 在 configs/ 目录下，新建目录 bevformer_vlm_align/ ， 在 configs/bevformer_vlm_align/ 目录下新建一个 bevformerv2-r50-t8-24ep_debertav3_align.py 的文件。它的内部结构通常是通过 _base_（继承机制） 把原有的配置吸纳进来，然后在一个全局的 model 字典里把三者拼接起来。
- BEVFormer-v2的配置文件是 projects/bevformerv2-r50-t8-24ep.py。
- DeBERTa-v3的配置文件是通过 Hugging Face 下载 “microsoft/deberta-v3-base” 相关。

## 约束
1. 1D Temporal PE 注入时切记要给 [CLS] 做 Padding（或跳过）
- 在将 [40, 768] 拼入 [CLS] 变成 [41, 768] 后注入 1D PE 时，代码通常是：x = x + self.temporal_pe。
- 注意你的 self.temporal_pe 必须是 [41, 768]。请确保第 0 位（即对应 [CLS] 的那一位位置编码）是一个固定的常数（如全零），或者是单独可学习的参数。不要让第 1 帧的位置编码错位加到了 [CLS] 上，否则会导致时序混乱。

2. Dataloader 侧的 Spatial PE 尺寸匹配
- 你的 BEV 稠密特征是 [40, 40000, 256]（对应 200 x 200 = 40000）。
- 在生成 2D Spatial PE 时，建议直接在 __init__ 中用 nn.Parameter 初始化一个 [200, 200, 256] 的可学习位置嵌入，在前向传播时用 view(1, 40000, 256) 展平，然后用 PyTorch 的广播机制直接加到 40 帧上：feat = feat + self.spatial_pe.view(1, 1, 40000, 256)。（feat 形状: [B, 40, 40000, 256]），这样可以保证时序上的每一帧都共享同一套标准的车体坐标系空间编码，这最符合自动驾驶 BEV 空间的物理规律。

# 程序兼容功能
为节省GPU资源，结合当前冻结部分网络参数的方式，程序做多种兼容模式的修改。程序执行时，可以通过传参进行模式选择。

## 原模式
原模式是指上述“训练方案架构描述-流程图1”的流程，继续保持。

## nuScenes数据集离线获取BEV特征模式
提供一种模式，在“训练方案架构描述-流程图1”里的“BEVFormer-v2”之后，“[40, 40000, 256] 稠密特征”之前，直接存储每一个BEV帧的特征。
- 通过BEVFormer-v2的推理，获得每一个BEV帧的特征，单独保存成pth格式。
- 这里的每一个BEV帧，指的是1个周期的多个camera环视组成的BEV，维度是[1, 40000, 256]，不是40帧一组的[40, 40000, 256]。
- 命名方式是“scene_number_token.pth”，其中scene来自nuScenes数据集中meta中的scene.json中的name；number是这个scene.json中的nbr_samples第几组，例如第1组number是0，依次累加；token来自scene.json中的token。
- 根据nuScenes数据集中meta中的scene.json中的nbr_samples, first_sample_token，last_sample_token，对应计算所有帧。
- 所有BEV特征的pth文件保存在一个文件夹中，路径可以记录在配置文件中。
- 在上述BEV特征的pth的文件夹中，保存一个bev_feature.json的文件，保存相关信息如下：
```json
[
    {
        "token": "",    //通过执行：openssl rand -hex 16，获取该值。
        "scene_token": "",  //来自nuScenes数据集中meta中的scene.json中的“token”。
        "frame_nbr": "", //这个scene的40个关键帧组的第几组，例如第1组number是0，依次累加。
        "frame_token": "",  //通过scene.json中first_sample_token->sample.json中next依次遍历找到。
        "filename": "", //对应文件的文件名，只显示文件名，不用显示绝对路径或相对路径。
    },
    {
        ......
    },
    ......
]
```
- 该模式只做BEV帧的特征提取，不继续执行后续的推理、训练、验证等步骤流程。

## nuScenes数据集离线训练模式
提供一种模式，在“训练方案架构描述-流程图1”里的“[40, 40000, 256] 稠密特征”之前，离线获取每一个BEV帧的特征，继续执行后续的训练步骤。注意：这里的每一个BEV帧，指的是1个周期的多个camera环视组成的BEV，维度是[1, 40000, 256]，不是40帧一组的[40, 40000, 256]。
- 数据来源是一个文件夹，路径可以记录在配置文件中。包含每一个BEV帧特征的pth文件，和一个总的bev_feature.json描述文件，定义参考“nuScenes数据集离线获取BEV特征模式”中的描述。

## nuScenes数据集离线推理模式
提供一种模式，同“nuScenes数据集离线训练模式”，在“训练方案架构描述-流程图1”里的“[40, 40000, 256] 稠密特征”之前，离线获取每一个BEV帧的特征，仅继续执行后续的推理步骤。注意：这里的每一个BEV帧，指的是1个周期的多个camera环视组成的BEV，维度是[1, 40000, 256]，不是40帧一组的[40, 40000, 256]。

## nuScenes数据集离线推理验证模式
提供一种模式，同“nuScenes数据集离线训练模式”，在“训练方案架构描述-流程图1”里的“[40, 40000, 256] 稠密特征”之前，离线获取每一个BEV帧的特征，仅继续执行后续的推理和验证步骤。注意：这里的每一个BEV帧，指的是1个周期的多个camera环视组成的BEV，维度是[1, 40000, 256]，不是40帧一组的[40, 40000, 256]。

# 训练效率
三大部分，即 BEVFormer-v2, DeBERTa-v3-base, lightweight transformer （对齐网络），在离线提取bev特征，训练、推理、验证时，使用PyTorch 的 AMP（Automatic Mixed Precision，自动混合精度）采用BF16精度。

## 采用“权重保持 FP32 + autocast”机制
- 初始化时，BEVFormer-v2、DeBERTa-v3-base、lightweight transformer 的权重保持原生 FP32，不做全模型 .to(torch.bfloat16) 强转。
- 在 forward 前向传播时，用 PyTorch 官方的 torch.cuda.amp.autocast(dtype=torch.bfloat16) 包裹，让 AMP 自动选择算子精度：
    - 支持 BF16 的算子走 BF16（节省显存、提升吞吐）。
    - 不支持 BF16 的算子（例如部分环境中的 nearest2d）自动回退 FP32，避免运行时报错。

## 区分“运行状态”与“落盘状态”
- BEVFormer-v2 和 DeBERTa-v3：因为在当前训练中是完全冻结（requires_grad=False）的，它们在硬盘上的官方预训练文件保持原生 FP32 格式，不需要改动。
- Lightweight Transformer （对齐网络）：是当前唯一需要更新参数和落盘的文件。在“FP32 权重 + AMP autocast”方案下，训练时计算图可混精，但参数落盘默认保持参数自身 dtype（通常为 FP32）。

## 小批量训练的负样本扩展（Feature Queue）
在多卡显存受限场景下（例如每卡物理 batch 较小），除了 DDP gather 和梯度累加，还可以引入特征队列来扩展对比学习负样本池。

### 当前实现版本（队列 + 可选 scene mask）
- 仅在训练阶段启用队列，验证阶段关闭队列，避免影响检索评估语义。
- 维护两个 FIFO 队列：
    - text_queue: 历史文本特征。文本侧入队的是 text_feat（形状 [B, 768]，每个 scene 文本 1 个向量）。
    - vision_queue: 历史视觉特征。视觉侧入队的是经过时序编码后得到的 vision_feat（形状 [B, 768]，每个样本 1 个向量，已聚合 40 帧信息）。
- 同时维护 scene id 队列：
    - text_queue_scene_id / vision_queue_scene_id（与特征队列同长度）。
- 当前 batch 仍按原有对称 InfoNCE 计算正样本；队列特征仅作为额外负样本拼接到 logits 右侧。
- 当 queue_use_scene_mask=True 时，会按 scene_id 对队列负样本做掩码：同 scene 的队列项不参与负样本竞争，降低假负样本干扰。
- 队列更新在每个训练 step 后执行，支持 DDP 下跨卡聚合后再入队（特征和 scene id 一起 gather）。
- 提供 warmup 机制，前若干 step 仅使用当前批全局负样本，队列填充到稳定后再参与 loss。

### 推荐起始参数
- use_feature_queue=True
- feature_queue_size=128
- feature_queue_warmup_steps=50
- gather_ddp=True
- queue_use_scene_mask=True

### 与梯度累加的配合
- 梯度累加用于提高“有效 batch”稳定性。
- 特征队列用于提高“负样本数量”。
- 两者互补，可同时开启。

### 边界与注意事项
- 队列特征是历史快照，存在一定陈旧性（staleness），因此不建议一开始把队列设得过大。
- scene mask 按“scene id 相同”掩码，不是按特征相似度阈值掩码。
- 若极端情况下某样本可用负样本过少（例如队列很小且场景高度重复），可临时关闭 scene mask 做对照。
- 队列特征必须 detach 后入队，不能反向回传到历史 step。

### 后续升级路径
- 第一步：先固定 queue_use_scene_mask 开关做 A/B 对照（True/False）。
- 第二步：结合显存与稳定性再调 feature_queue_size 与 warmup。
- 第三步：视资源情况叠加更大累加步数或更严格负样本策略。

## DDP 启动稳定性补充（2026-06-22）
在 offline_train + DDP + DataLoader 多进程场景下，若遇到：
- TypeError: cannot pickle 'dict_keys' object

已知根因是数据集中 eval_detection_configs.class_names 在部分环境下是 dict_keys 视图对象，spawn worker 时不可序列化。

当前修复策略：
- 在 CustomNuScenesDatasetV2 初始化阶段统一把该字段转换为 list。

效果：
- DDP 能正常越过 DataLoader worker 启动并进入反向传播。

## 多机多卡模式

### 只同步多机共同训练的必要文件
- 每个 epoch 仅同步 epoch_${epoch}.pth（这是下一轮 resume 必需文件）。
- 默认不同步 align_trainable_epoch_*.pth / latest.pth。

### 使用 rsync password-file（daemon 模式）支持多机串行多轮 epoch
- 在脚本内加入自动同步逻辑（仅 rank0 执行）。
- 非 rank0 节点在下一轮 resume 前会等待断点文件到位，避免“文件未同步就报错”。

## 串行逐 epoch 训练与验证（2026-07-08）

当前推荐使用 `tools/train_val_epoch_serial_ddp.sh` 做 trainval 长跑：每次启动只完成一个目标 epoch，随后立刻跑验证、写入 metrics，再进入下一轮 resume。

### LR 策略
- `SERIAL_LR_POLICY=config`: 使用配置文件内的 LR scheduler。此时 `SERIAL_TOTAL_EPOCHS` 可以作为 cosine horizon，脚本通过 stop hook 保证单次 train command 仍只跑完当前目标 epoch。
- `SERIAL_LR_POLICY=Fixed`: 使用固定 LR。若设置 `SERIAL_FIXED_LR`，脚本会同时传入 `optimizer.lr` 和 `serial_resume_optimizer_lr`，避免 resume checkpoint 后 optimizer param_groups 恢复旧 LR。

### StopAfterTargetEpochHook
- 脚本会按需传入 `serial_stop_after_epoch=<epoch>`。
- `tools/train.py` 在读取配置后注入 `StopAfterTargetEpochHook`。
- hook 在目标 epoch 完成后把 runner 的 `_max_epochs` clamp 到当前完成 epoch，从而让每次训练启动只前进一轮。

### Resume 结构兼容
- 若模型结构变化导致 checkpoint 中 optimizer state 的 parameter group 数量与当前模型不一致，训练入口会 fallback 到只加载模型权重和 runner epoch/iter 状态，不恢复 optimizer state。
- 该 fallback 适用于新增 NetVLAD、Temporal Attention Pooling 等 trainable 参数后的继续训练。
- fallback 后仍会应用 `serial_resume_optimizer_lr`，保证固定 LR 模式不会被旧 checkpoint 覆盖。

### 多节点一致性检查
- 串行脚本会记录并校验 `START_EPOCH`、`END_EPOCH`、`SERIAL_TOTAL_EPOCHS`、`SERIAL_LR_POLICY`、`SERIAL_FIXED_LR`、`DATASET_PROFILE`、`RDZV_ID`。
- 同时记录 config、detector 文件、`tools/train.py` 的 sha256 指纹。
- 若两台机器代码或配置不一致，脚本会在启动阶段报错，避免 DDP 运行到参数数量不一致后才失败。

## Embedding Diagnostics（2026-07-08）

当前验证脚本支持导出 embedding 诊断 JSON，用于判断模型是否仍处于 collapse 或 hubness 状态。

### 启用方式
- 串行脚本中设置 `EMBEDDING_DIAGNOSTICS_ENABLE=true`。
- 建议同时设置 `DIAGNOSTICS_SUBSET_ONE_PER_SCENE=true`，脚本会生成并复用 `WORK_DIR/tmp_val_one_per_scene.pkl`，每个 scene 只取一个样本，避免重复 clip 影响几何统计。
- 也可直接调用 `tools/validate_vlm_align.py --dump-embedding-diagnostics <path>`。

### 导出内容
- i2t rank、positive/negative similarity、max negative similarity、margin、margin_positive_ratio。
- text-text 和 vision-vision 的 offdiag similarity，用于观察两侧 embedding 是否过度同向。
- feature geometry：centroid norm、cosine_to_centroid、centered_l2_norm、PCA explained variance、effective rank。
- hubness：i2t top1 scene frequency、max negative scene frequency、t2i top1 scene frequency。
- worst cases：margin 最差的样本、真值 scene、预测 scene、分数和 rank。
- layers：当模型打开 `return_intermediate_feats` 时，导出 `vision_spatial_pooled`、`vision_temporal_cls`、`vision_temporal_frame_mean`、`vision_temporal_attn`、`vision_netvlad`、`vision_pre_norm`、`vision_projector_out`、`vision_final`、`text_raw_pool`、`text_projector_out`、`text_final` 等中间层统计。

### 常用判断
- 若 `vision_vision.offdiag_similarity` 接近 1，说明视觉侧仍高度坍缩。
- 若 `text_text.offdiag_similarity` 明显低于 vision，而 vision 接近 1，优先检查视觉池化和视觉正则。
- 若 `hubness.i2t_top1_scene_frequency.max_ratio` 很高，说明大量 query 被同一个文本吸走，需要关注 common direction 和负样本构成。
- 若 `layers.vision_netvlad` 已经分散但 `vision_final` 坍缩，优先检查 projector 或 normalize 前后的特征。


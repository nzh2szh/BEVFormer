_base_ = ['../../projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Use real 40-frame temporal input (-39 ... 0) instead of inheriting t8 frames.
frames = tuple(range(-39, 1))

model = dict(
    type='BEVFormerDebertaAlign',
    frames=frames,
    # disable mono branch for alignment-only training
    fcos3d_bbox_head=None,
    mono_loss_weight=0.0,
    text_model_name='microsoft/deberta-v3-base',
    # Pin to a fixed Hugging Face snapshot for reproducible runs.
    text_model_revision='8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
    # Set True to force local-only load (no network) after model is cached.
    text_model_local_files_only=False,
    # Keep Hugging Face cache inside project ckpts directory.
    text_model_cache_dir='ckpts/hf_cache',
    scene_json='data/nuscenes/v1.0-trainval/scene.json',
    scene_text_field='description',
    # compat mode: origin|offline_extract_bev|offline_train|offline_infer|offline_infer_validate
    # aliases are supported internally: origin->online, offline_extract_bev->extract
    run_mode='origin',
    offline_bev_dir='data/nuscenes/bev_offline_features',
    # Optional split-aware directories. If provided, runtime picks split by mode:
    # offline_train->train, offline_infer_validate->val, offline_infer->test.
    offline_bev_dir_by_split=dict(
        train='data/nuscenes/bev_offline_features_train',
        val='data/nuscenes/bev_offline_features_val',
        test='data/nuscenes/bev_offline_features_test',
    ),
    offline_metadata_name='bev_feature.json',
    offline_dump_overwrite=False,
    expected_frames=40,
    max_frames=40,
    bev_embed_dims=256,
    proj_dims=768,
    temporal_encoder_layers=3,
    temporal_num_heads=8,
    temporal_ffn_dims=2048,
    spatial_bev_h=200,
    spatial_bev_w=200,
    gather_ddp=True,
    # Enable BF16 runtime for frozen backbones and alignment head.
    use_bf16_amp=True,
    dropout=0.1,
    temperature=0.07,
)

# Alignment task only uses retrieval loss.
find_unused_parameters = True

# Keep backbone ckpt loading for frozen BEVFormer weights.
load_from = './ckpts/epoch_24.pth'

# Raise batch size as much as memory allows to improve in-batch negatives.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(frames=frames),
    val=dict(frames=frames),
    test=dict(frames=frames),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-2,
)

total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Save only alignment trainable weights, not full model checkpoints.
checkpoint_config = None
custom_hooks = [
    dict(
        type='SaveTrainableStateDictHook',
        by_epoch=True,
        interval=1,
        filename_tmpl='align_trainable_epoch_{}.pth',
    )
]

# Use no-validate in tools/train.py, or provide a custom retrieval evaluator.
evaluation = dict(interval=1000)

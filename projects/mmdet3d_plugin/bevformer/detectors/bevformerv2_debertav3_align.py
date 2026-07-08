import json
import bisect
import hashlib
import math
import os
import secrets
import warnings
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS

from .bevformerV2 import BEVFormerV2


@DETECTORS.register_module()
class BEVFormerDebertaAlign(BEVFormerV2):
    """Frozen BEVFormer + frozen DeBERTa-v3 with trainable alignment heads.

    The model keeps both backbones frozen and only optimizes newly added
     alignment modules: spatial pooling, temporal encoder, and projectors.

     High-level data flow:
     1) Vision tower:
         - Extract dense BEV tokens from frozen BEVFormer.
         - Inject 2D spatial positional embedding.
         - Pool each frame from [HW, C] to [1, C] with a learnable query.
         - Upsample to 768 dims, prepend CLS, inject 1D temporal PE.
         - Fuse with lightweight temporal Transformer and take CLS.
         - Project and L2-normalize.

     2) Text tower:
         - Encode sentence with frozen DeBERTa.
         - Masked mean pooling to remove PAD effect.
         - Project and L2-normalize.

     3) Cross-modal objective:
         - Compute symmetric InfoNCE (i2t + t2i).
         - Optionally gather features across DDP workers to increase negatives.
    """

    def __init__(self,
                 text_model_name='microsoft/deberta-v3-base',
                 text_model_revision=None,
                 text_model_local_files_only=False,
                 text_model_cache_dir=None,
                 text_max_length=512,
                 scene_json='data/nuscenes/v1.0-trainval/scene.json',
                 scene_text_field='description',
                 expected_frames=40,
                 bev_embed_dims=256,
                 proj_dims=768,
                 temporal_encoder_layers=3,
                 temporal_num_heads=8,
                 temporal_ffn_dims=2048,
                 dropout=0.1,
                 temperature=0.07,
                 max_frames=40,
                 spatial_bev_h=200,
                 spatial_bev_w=200,
                 vision_pooling='cls',
                 vision_netvlad_clusters=8,
                 use_vision_projector=True,
                 vision_projector_residual_weight=1.0,
                 use_text_projector=True,
                 vision_batch_centering_weight=0.0,
                 vision_covariance_reg_weight=0.0,
                 gather_ddp=True,
                 use_feature_queue=False,
                 feature_queue_size=128,
                 feature_queue_warmup_steps=50,
                 queue_use_scene_mask=False,
                 use_multi_positive=True,
                 use_bf16_amp=True,
                 run_mode='online',
                 offline_bev_dir='',
                 offline_bev_dir_by_split=None,
                 offline_split=None,
                 offline_metadata_name='bev_feature.json',
                 offline_dump_overwrite=False,
                 offline_sequence_mode='past_then_future',
                 **kwargs):
        """Initialize BEVFormer-DeBERTa alignment model.

        Args:
            text_model_name (str): Hugging Face model name/path for the frozen
                text encoder (e.g. microsoft/deberta-v3-base).
            text_model_revision (str | None): Optional Hugging Face revision
                (branch/tag/commit hash) for reproducible text model loading.
            text_model_local_files_only (bool): If True, only load text model
                files from local cache/path and never access network.
            text_model_cache_dir (str | None): Optional cache directory passed
                to Hugging Face `from_pretrained`.
            text_max_length (int): Maximum token length for scene text
                encoding. Longer descriptions are truncated before DeBERTa.
            scene_json (str): Path to scene metadata file used to map
                scene_token -> text description.
            scene_text_field (str): Field name in scene.json used as text
                target. Falls back to `name` when empty.
            expected_frames (int): Target temporal length for each sample.
                Input clips are sampled/padded to this length.
            bev_embed_dims (int): Channel size of BEVFormer dense BEV token
                features before alignment projection (typically 256).
            proj_dims (int): Shared embedding dimension for vision/text
                alignment space (typically 768).
            temporal_encoder_layers (int): Number of layers in lightweight
                temporal Transformer encoder.
            temporal_num_heads (int): Attention heads in temporal encoder.
            temporal_ffn_dims (int): FFN hidden size in temporal encoder.
            dropout (float): Dropout ratio used by attention/encoder blocks.
            temperature (float): Initial InfoNCE temperature. Stored as log
                scale in `logit_scale` for stable optimization.
            max_frames (int): Maximum temporal positions to allocate for 1D
                temporal positional embeddings.
            spatial_bev_h (int): BEV height used by 2D spatial positional
                embedding.
            spatial_bev_w (int): BEV width used by 2D spatial positional
                embedding.
            vision_pooling (str): Temporal aggregation strategy after the
                spatial pooling. `cls` uses the temporal CLS token;
                `frame_mean` uses the mean of encoded frame tokens;
                `spatial_mean` uses the mean of pre-temporal frame tokens;
                `temporal_attn` uses one learnable temporal query over encoded
                frame tokens; `netvlad` uses multi-cluster residual pooling
                over encoded frame tokens.
            vision_netvlad_clusters (int): Number of NetVLAD clusters used
                when vision_pooling='netvlad'.
            use_vision_projector (bool): Whether to pass vision features
                through the trainable vision projector before normalization.
            vision_projector_residual_weight (float): Interpolation strength
                for the vision projector path. 1.0 uses projector output
                directly; smaller values preserve more raw vision feature.
            use_text_projector (bool): Whether to pass frozen text features
                through the trainable text projector before normalization.
            vision_batch_centering_weight (float): Weight for penalizing the
                squared norm of the gathered vision-feature batch mean. This
                discourages hubness caused by a dominant common direction.
            vision_covariance_reg_weight (float): Weight for off-diagonal
                covariance regularization on gathered vision features.
            gather_ddp (bool): Whether to all-gather features across DDP
                workers when computing contrastive loss.
            use_feature_queue (bool): Whether to use FIFO feature queues as
                extra negatives for contrastive loss in training.
            feature_queue_size (int): Queue size per modality.
            feature_queue_warmup_steps (int): Number of global train steps
                before queue negatives are used in loss.
            queue_use_scene_mask (bool): Whether to mask queue negatives from
                the same scene as current sample when computing logits.
            use_multi_positive (bool): Whether to optimize multi-positive
                InfoNCE where samples sharing scene/text are treated as
                positives instead of negatives.
                        run_mode (str): Runtime mode for compatibility scenarios.
                                Supported values:
                                - origin: original mode, extract BEV features online.
                                    compatiblity of alias online.
                                - offline_extract_bev: extract and dump BEV features only.
                                    compatiblity of alias extract.
                                - offline_train: load BEV features from offline directory
                                    before training head.
                                - offline_infer: load BEV features from offline directory
                                    before inference.
                                - offline_infer_validate: same as offline_infer for
                                    inference+validation workflow.
                        offline_bev_dir (str): Directory containing dumped BEV feature
                                pth files and metadata json.
                        offline_bev_dir_by_split (dict | None): Optional split-aware
                            directories, e.g. {'train': '...', 'val': '...',
                            'test': '...'}. When provided, it overrides
                            `offline_bev_dir` for the active split.
                        offline_split (str | None): Active split key used with
                            `offline_bev_dir_by_split`. If None, a default
                            split is chosen from run_mode.
                        offline_metadata_name (str): Metadata json filename under
                                offline_bev_dir.
                        offline_dump_overwrite (bool): Whether overwrite existed pth/json
                                when dumping BEV features in extract mode.
                        offline_sequence_mode (str): Policy for building
                            offline temporal clips.
                            - past_only: use only frames <= anchor.
                            - past_then_future: use past first; if shorter
                              than expected_frames, append later frames in
                              the same scene before final padding.
            **kwargs: Remaining BEVFormerV2 initialization arguments.
        """
        super().__init__(**kwargs)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'transformers is required for BEVFormerDebertaAlign. '
                'Please install it with: pip install transformers'
            ) from exc

        self.expected_frames = expected_frames
        self.max_frames = max(max_frames, expected_frames)
        self.text_max_length = int(text_max_length)
        self.scene_json = scene_json
        self.scene_text_field = scene_text_field
        self.spatial_bev_h = spatial_bev_h
        self.spatial_bev_w = spatial_bev_w
        valid_vision_pooling = {'cls', 'frame_mean', 'spatial_mean', 'temporal_attn', 'netvlad'}
        if vision_pooling not in valid_vision_pooling:
            raise ValueError(
                'Unsupported vision_pooling: {}. valid values are {}.'.format(
                    vision_pooling,
                    sorted(valid_vision_pooling),
                )
            )
        self.vision_pooling = vision_pooling
        self.vision_netvlad_clusters = int(vision_netvlad_clusters)
        if self.vision_netvlad_clusters <= 0:
            raise ValueError(
                'vision_netvlad_clusters must be > 0, got {}.'.format(
                    self.vision_netvlad_clusters,
                )
            )
        self.use_vision_projector = use_vision_projector
        self.vision_projector_residual_weight = float(vision_projector_residual_weight)
        self.use_text_projector = use_text_projector
        self.vision_batch_centering_weight = float(vision_batch_centering_weight)
        self.vision_covariance_reg_weight = float(vision_covariance_reg_weight)
        self.gather_ddp = gather_ddp
        self.use_feature_queue = use_feature_queue
        self.feature_queue_size = int(feature_queue_size)
        self.feature_queue_warmup_steps = int(feature_queue_warmup_steps)
        self.queue_use_scene_mask = queue_use_scene_mask
        self.use_multi_positive = use_multi_positive
        self.use_bf16_amp = use_bf16_amp
        self.run_mode = run_mode
        self.offline_bev_dir = offline_bev_dir
        self.offline_bev_dir_by_split = offline_bev_dir_by_split or {}
        self.offline_split = offline_split
        self.offline_metadata_name = offline_metadata_name
        self.offline_dump_overwrite = offline_dump_overwrite
        self.offline_sequence_mode = offline_sequence_mode

        mode_alias = {
            'origin': 'online',
            'offline_extract_bev': 'extract',
        }
        self.run_mode = mode_alias.get(self.run_mode, self.run_mode)

        self._valid_run_modes = {
            'online',
            'extract',
            'offline_train',
            'offline_infer',
            'offline_infer_validate',
        }
        if self.run_mode not in self._valid_run_modes:
            raise ValueError(
                'Unsupported run_mode: {}. valid modes are {}.'.format(
                    self.run_mode,
                    sorted(self._valid_run_modes),
                )
            )
        if self.offline_sequence_mode not in {'past_only', 'past_then_future'}:
            raise ValueError(
                'Unsupported offline_sequence_mode: {}. valid values are {}.'.format(
                    self.offline_sequence_mode,
                    ['past_only', 'past_then_future'],
                )
            )

        if not isinstance(self.offline_bev_dir_by_split, dict):
            raise TypeError('offline_bev_dir_by_split must be a dict when provided.')
        if self.offline_split is None:
            self.offline_split = self._default_offline_split()

        self._offline_records = []
        self._offline_record_index = {}
        self._offline_scene_frames = {}
        if self.run_mode in {'offline_train', 'offline_infer', 'offline_infer_validate'}:
            self._load_offline_bev_index()

        text_load_kwargs = {}
        if text_model_revision is not None:
            text_load_kwargs['revision'] = text_model_revision
        if text_model_local_files_only:
            text_load_kwargs['local_files_only'] = True
        if text_model_cache_dir is not None:
            text_load_kwargs['cache_dir'] = text_model_cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, **text_load_kwargs)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, **text_load_kwargs)
        self._freeze_module(self.text_encoder)

        # Learnable 2D BEV positional embedding shared across all frames.
        self.spatial_pe = nn.Parameter(
            torch.zeros(self.spatial_bev_h, self.spatial_bev_w, bev_embed_dims)
        )
        # A single global query pools dense BEV tokens into one token per frame.
        self.spatial_query = nn.Parameter(torch.randn(1, 1, bev_embed_dims))
        self.spatial_pool = nn.MultiheadAttention(
            embed_dim=bev_embed_dims,
            num_heads=8,
            dropout=dropout,
            batch_first=True)
        self.vision_up_proj = nn.Linear(bev_embed_dims, proj_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dims))
        # Separate CLS temporal PE prevents frame-1 PE from being applied to CLS.
        self.cls_temporal_pe = nn.Parameter(torch.zeros(1, 1, proj_dims))
        self.temporal_pe = nn.Parameter(torch.zeros(1, self.max_frames, proj_dims))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dims,
            nhead=temporal_num_heads,
            dim_feedforward=temporal_ffn_dims,
            dropout=dropout,
            batch_first=True,
            activation='gelu')
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=temporal_encoder_layers)
        self.temporal_pool_query = nn.Parameter(torch.zeros(1, 1, proj_dims))
        self.temporal_pool = nn.MultiheadAttention(
            embed_dim=proj_dims,
            num_heads=temporal_num_heads,
            dropout=dropout,
            batch_first=True)
        self.vision_netvlad_assign = nn.Linear(proj_dims, self.vision_netvlad_clusters)
        self.vision_netvlad_clusters_param = nn.Parameter(
            torch.randn(self.vision_netvlad_clusters, proj_dims) * 0.02
        )
        self.vision_netvlad_proj = nn.Linear(self.vision_netvlad_clusters * proj_dims, proj_dims)

        self.vision_projector = nn.Sequential(
            nn.Linear(proj_dims, proj_dims),
            nn.GELU(),
            nn.Linear(proj_dims, proj_dims))

        self.text_projector = nn.Sequential(
            nn.Linear(proj_dims, proj_dims),
            nn.GELU(),
            nn.Linear(proj_dims, proj_dims))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

        if self.feature_queue_size <= 0:
            raise ValueError('feature_queue_size must be > 0, got {}.'.format(self.feature_queue_size))
        # Queues store normalized projector outputs as detached negatives.
        self.register_buffer('text_queue', torch.zeros(self.feature_queue_size, proj_dims))
        self.register_buffer('vision_queue', torch.zeros(self.feature_queue_size, proj_dims))
        self.register_buffer('text_queue_scene_id', torch.full((self.feature_queue_size,), -1, dtype=torch.long))
        self.register_buffer('vision_queue_scene_id', torch.full((self.feature_queue_size,), -1, dtype=torch.long))
        self.register_buffer('text_queue_text_id', torch.full((self.feature_queue_size,), -1, dtype=torch.long))
        self.register_buffer('vision_queue_text_id', torch.full((self.feature_queue_size,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_valid_len', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_step', torch.zeros(1, dtype=torch.long))

        self.scene_token_to_text = self._load_scene_text_map()
        self.scene_token_to_name = self._load_scene_name_map()
        self.scene_sample_order = self._load_scene_sample_order_map()
        self._freeze_visual_backbone()
        # Keep all module weights in native FP32. AMP autocast controls runtime
        # precision and automatically falls back for unsupported BF16 operators.

    def _amp_autocast_context(self):
        """Return BF16 autocast context when CUDA runtime is available."""
        if self.use_bf16_amp and torch.cuda.is_available():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return nullcontext()

    def _bevformer_safe_context(self):
        """Run BEVFormer extraction in FP32 to bypass unsupported BF16 ops."""
        if self.use_bf16_amp and torch.cuda.is_available():
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _freeze_module(self, module):
        """Freeze a module and force eval mode."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_visual_backbone(self):
        """Keep BEVFormer branches frozen during alignment training.

        These branches are reused only as feature extractors and should not
        receive gradient updates in this alignment task.
        """
        freeze_targets = [
            self.img_backbone,
            self.img_neck,
            self.pts_bbox_head,
            self.fcos3d_bbox_head,
        ]
        for module in freeze_targets:
            if module is not None:
                self._freeze_module(module)

    def train(self, mode=True):
        """Override train() to keep frozen towers in eval/no-grad behavior.

        Even when the full model enters training mode, frozen backbones are
        forced back to eval mode to avoid behavior drift (e.g., BN updates).
        """
        super().train(mode)
        self._freeze_visual_backbone()
        self.text_encoder.eval()
        return self

    def _load_scene_text_map(self):
        """Load scene token -> text description mapping from scene.json.

        Falls back to scene name when the configured text field is empty.
        Returns an empty map if the file does not exist.
        """
        text_map = {}
        if not self.scene_json:
            return text_map
        if not os.path.isfile(self.scene_json):
            return text_map

        with open(self.scene_json, 'r', encoding='utf-8') as f:
            scene_records = json.load(f)

        for record in scene_records:
            token = record.get('token', '')
            desc = record.get(self.scene_text_field, '')
            if not desc:
                desc = record.get('name', '')
            if token:
                text_map[token] = desc
        return text_map

    def _load_scene_name_map(self):
        """Load scene token -> scene name mapping from scene.json."""
        name_map = {}
        if not self.scene_json:
            return name_map
        if not os.path.isfile(self.scene_json):
            return name_map

        with open(self.scene_json, 'r', encoding='utf-8') as f:
            scene_records = json.load(f)

        for record in scene_records:
            token = record.get('token', '')
            name = record.get('name', '')
            if token and name:
                name_map[token] = name
        return name_map

    def _load_scene_sample_order_map(self):
        """Build scene-token + sample-token -> sequential frame index map.

        The order is derived from scene.json (first/last/nbr_samples) and
        sample.json linked-list traversal, matching nuScenes metadata.
        """
        order_map = {}
        if not self.scene_json or (not os.path.isfile(self.scene_json)):
            raise FileNotFoundError('scene.json not found: {}'.format(self.scene_json))

        sample_json = os.path.join(os.path.dirname(self.scene_json), 'sample.json')
        if not os.path.isfile(sample_json):
            raise FileNotFoundError('sample.json not found next to scene.json: {}'.format(sample_json))

        with open(self.scene_json, 'r', encoding='utf-8') as f:
            scene_records = json.load(f)
        with open(sample_json, 'r', encoding='utf-8') as f:
            sample_records = json.load(f)

        sample_by_token = {}
        for rec in sample_records:
            token = rec.get('token', '') if isinstance(rec, dict) else ''
            if token:
                sample_by_token[token] = rec

        for scene in scene_records:
            if not isinstance(scene, dict):
                continue
            scene_token = scene.get('token', '')
            first_token = scene.get('first_sample_token', '')
            last_token = scene.get('last_sample_token', '')
            nbr_samples = scene.get('nbr_samples', None)
            if not scene_token or not first_token:
                continue

            idx = 0
            token = first_token
            visited = set()
            while token:
                if token in visited:
                    warnings.warn(
                        'Detected sample token loop in scene {} at token {}'.format(scene_token, token)
                    )
                    break
                visited.add(token)

                rec = sample_by_token.get(token)
                if rec is None:
                    warnings.warn(
                        'sample token {} from scene {} is missing in sample.json'.format(token, scene_token)
                    )
                    break

                order_map['{}::{}'.format(scene_token, token)] = idx
                idx += 1

                if token == last_token:
                    break
                token = rec.get('next', '')

            if isinstance(nbr_samples, int) and nbr_samples > 0 and idx != nbr_samples:
                warnings.warn(
                    'scene {} sample count mismatch: traversed {} vs nbr_samples {}'.format(
                        scene_token, idx, nbr_samples)
                )

        return order_map

    def _parse_temporal_meta(self, img_metas):
        """Sort temporal keys so frame order is deterministic.

        Args:
            img_metas: list[dict], each dict contains temporal entries keyed
                by frame index (or frame-like key).
        Returns:
            list[OrderedDict]: one ordered temporal meta map per sample.
        """
        temporal_metas = []
        for sample_meta in img_metas:
            ordered = OrderedDict(sorted(sample_meta.items(), key=lambda x: x[0]))
            temporal_metas.append(ordered)
        return temporal_metas

    def _get_anchor_meta(self, sample_metas):
        """Pick anchor frame meta for naming and indexing.

        Priority uses key 0 (current frame). If absent, use largest key as the
        nearest-to-current fallback.
        """
        if 0 in sample_metas:
            return sample_metas[0]
        last_key = max(sample_metas.keys())
        return sample_metas[last_key]

    def _get_scene_token_and_group(self, sample_metas):
        """Get (scene_token, frame_nbr, frame_token) from temporal metadata."""
        anchor_meta = self._get_anchor_meta(sample_metas)
        scene_token = anchor_meta.get('scene_token', '')
        sample_token = anchor_meta.get('sample_idx', '')
        if not scene_token:
            raise KeyError('scene_token is missing in img_metas anchor frame.')
        if not sample_token:
            raise KeyError('sample_idx(frame token) is missing in img_metas anchor frame.')

        order_key = '{}::{}'.format(scene_token, sample_token)
        if order_key not in self.scene_sample_order:
            raise KeyError(
                'frame token {} is not found in scene {} sample chain built from scene.json/sample.json'.format(
                    sample_token, scene_token)
            )
        return scene_token, int(self.scene_sample_order[order_key]), sample_token

    def _sanitize_filename_part(self, text):
        if text is None:
            return ''
        return str(text).replace('/', '_').replace(' ', '_')

    def _offline_feature_filename(self, scene_name, scene_token, keyframe_nbr):
        """Build feature filename as scene_number_token.pth style."""
        safe_scene = self._sanitize_filename_part(scene_name) or 'unknown_scene'
        safe_token = self._sanitize_filename_part(scene_token) or 'unknown_token'
        return '{}_{}_{}.pth'.format(safe_scene, keyframe_nbr, safe_token)

    def _offline_metadata_path(self):
        active_dir = self._get_active_offline_bev_dir()
        if not active_dir:
            return ''
        return os.path.join(active_dir, self.offline_metadata_name)

    def _default_offline_split(self):
        if self.run_mode == 'offline_train':
            return 'train'
        if self.run_mode == 'offline_infer_validate':
            return 'val'
        if self.run_mode == 'offline_infer':
            return 'test'
        if self.run_mode == 'extract':
            return 'train'
        return None

    def _get_active_offline_bev_dir(self):
        if self.offline_split and self.offline_split in self.offline_bev_dir_by_split:
            split_dir = self.offline_bev_dir_by_split.get(self.offline_split)
            if split_dir:
                return split_dir
        return self.offline_bev_dir

    def _is_rank0(self):
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _load_offline_bev_index(self):
        """Load bev_feature.json into fast index for offline read mode."""
        active_dir = self._get_active_offline_bev_dir()
        meta_path = self._offline_metadata_path()
        if not active_dir:
            raise ValueError('offline_bev_dir is required for offline_* modes.')
        if not os.path.isdir(active_dir):
            raise FileNotFoundError(
                'offline_bev_dir does not exist: {}'.format(active_dir)
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                'Offline metadata file not found: {}'.format(meta_path)
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError('Offline metadata must be a JSON list.')

        self._offline_records = records
        self._offline_record_index = {}
        scene_frame_set = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            scene_token = rec.get('scene_token', '')
            frame_nbr = rec.get('frame_nbr', '')
            if scene_token == '' or frame_nbr == '':
                continue
            try:
                key = '{}::{}'.format(scene_token, int(frame_nbr))
            except (TypeError, ValueError):
                continue
            self._offline_record_index[key] = rec
            if scene_token not in scene_frame_set:
                scene_frame_set[scene_token] = set()
            scene_frame_set[scene_token].add(int(frame_nbr))

        self._offline_scene_frames = {
            scene: sorted(frame_set)
            for scene, frame_set in scene_frame_set.items()
        }

    def _select_offline_frame_numbers(self, scene_token, anchor_frame_nbr):
        """Select ordered frame numbers for one offline temporal clip."""
        scene_frames = self._offline_scene_frames.get(scene_token, [])
        if len(scene_frames) == 0:
            return []

        anchor = int(anchor_frame_nbr)
        end = bisect.bisect_right(scene_frames, anchor)
        if end <= 0:
            end = 1

        if self.offline_sequence_mode == 'past_only':
            start = max(0, end - self.expected_frames)
            return scene_frames[start:end]

        # past_then_future: keep history priority, then append newer frames
        # when history is short at early scene anchors.
        start = max(0, end - self.expected_frames)
        if end - start < self.expected_frames:
            need = self.expected_frames - (end - start)
            end = min(len(scene_frames), end + need)
        if end - start > self.expected_frames:
            start = end - self.expected_frames
        return scene_frames[start:end]

    def _dump_bev_features(self, bev_seq, img_metas):
        """Dump each sample BEV feature to pth and update bev_feature.json."""
        active_dir = self._get_active_offline_bev_dir()
        if not active_dir:
            raise ValueError('offline_bev_dir is required for extract mode.')
        if not self._is_rank0():
            return

        os.makedirs(active_dir, exist_ok=True)

        temporal_metas = self._parse_temporal_meta(img_metas)
        updated = False

        for b, sample_metas in enumerate(temporal_metas):
            scene_token, frame_nbr, frame_token = self._get_scene_token_and_group(sample_metas)
            if not scene_token:
                continue

            frame_keys = list(sample_metas.keys())
            anchor_key = 0 if 0 in sample_metas else max(sample_metas.keys())
            if anchor_key not in frame_keys:
                raise KeyError('Anchor key {} is not found in temporal metadata keys.'.format(anchor_key))
            anchor_idx = frame_keys.index(anchor_key)
            if anchor_idx >= bev_seq.shape[1]:
                raise IndexError(
                    'Anchor index {} out of range for BEV sequence length {}.'.format(
                        anchor_idx, bev_seq.shape[1])
                )

            anchor_meta = sample_metas[anchor_key]
            anchor_frame_token = anchor_meta.get('sample_idx', '')
            if anchor_frame_token != frame_token:
                raise ValueError(
                    'Frame token mismatch between mapping and anchor meta: mapped={}, anchor={}'.format(
                        frame_token, anchor_frame_token)
                )

            scene_name = anchor_meta.get('scene_name', '')
            if not scene_name:
                scene_name = self.scene_token_to_name.get(scene_token, '')

            filename = self._offline_feature_filename(scene_name, scene_token, frame_nbr)
            filepath = os.path.join(active_dir, filename)
            if (not os.path.isfile(filepath)) or self.offline_dump_overwrite:
                # Dump one BEV frame [1, HW, C] by exact anchor frame index.
                single_bev = bev_seq[b, anchor_idx:anchor_idx + 1, :, :].detach().cpu()
                if self.use_bf16_amp:
                    # Keep runtime extraction stable in FP32, but store offline
                    # features in BF16 to reduce disk size and I/O.
                    single_bev = single_bev.to(torch.bfloat16)
                torch.save(single_bev, filepath)

            key = '{}::{}'.format(scene_token, int(frame_nbr))
            prev_rec = self._offline_record_index.get(key, {})
            if isinstance(prev_rec, dict):
                record_token = prev_rec.get('token', '')
            else:
                record_token = ''
            if not record_token:
                # Keep the same semantics as `openssl rand -hex 16`.
                record_token = secrets.token_hex(16)

            rec = {
                'token': record_token,
                'scene_token': scene_token,
                'frame_nbr': int(frame_nbr),
                'frame_token': frame_token,
                'filename': filename,
            }
            self._offline_record_index[key] = rec
            updated = True

        if updated:
            records = list(self._offline_record_index.values())
            records.sort(key=lambda x: (x.get('scene_token', ''), int(x.get('frame_nbr', 0))))
            meta_path = self._offline_metadata_path()
            with open(meta_path, 'w', encoding='utf-8') as wf:
                json.dump(records, wf, ensure_ascii=False, indent=2)
            self._offline_records = records

    def _load_offline_bev_sequence(self, img_metas, device):
        """Load BEV sequence tensor [B, T, HW, C] from offline pth files."""
        temporal_metas = self._parse_temporal_meta(img_metas)
        batch_bev = []
        for sample_metas in temporal_metas:
            scene_token, frame_nbr, _ = self._get_scene_token_and_group(sample_metas)
            anchor_meta = self._get_anchor_meta(sample_metas)
            explicit_frame_nbrs = anchor_meta.get('offline_frame_nbrs', None)
            use_explicit_frame_nbrs = isinstance(explicit_frame_nbrs, (list, tuple)) and len(explicit_frame_nbrs) > 0
            if isinstance(explicit_frame_nbrs, (list, tuple)) and len(explicit_frame_nbrs) > 0:
                frame_numbers = [int(x) for x in explicit_frame_nbrs]
            else:
                frame_numbers = self._select_offline_frame_numbers(scene_token, frame_nbr)
            clip_frames = []
            missing_frame_numbers = []
            for hist_nbr in frame_numbers:
                key = '{}::{}'.format(scene_token, hist_nbr)
                record = self._offline_record_index.get(key)
                if record is None:
                    missing_frame_numbers.append(hist_nbr)
                    continue
                filename = record.get('filename', '')
                if not filename:
                    missing_frame_numbers.append(hist_nbr)
                    continue
                active_dir = self._get_active_offline_bev_dir()
                filepath = os.path.join(active_dir, filename)
                if not os.path.isfile(filepath):
                    missing_frame_numbers.append(hist_nbr)
                    continue

                loaded = torch.load(filepath, map_location='cpu')
                if isinstance(loaded, dict):
                    if 'bev_feature' in loaded:
                        loaded = loaded['bev_feature']
                    elif 'state_dict' in loaded and isinstance(loaded['state_dict'], torch.Tensor):
                        loaded = loaded['state_dict']
                    else:
                        raise ValueError('Unsupported offline BEV pth payload format: {}'.format(filepath))

                if loaded.dim() != 3:
                    raise ValueError(
                        'Offline BEV feature must be a single-frame tensor [1, HW, C], got {} from {}'.format(
                            tuple(loaded.shape), filepath)
                    )
                if loaded.shape[0] != 1:
                    raise ValueError(
                        'Legacy multi-frame offline BEV is no longer supported. '
                        'Expected [1, HW, C], got {} from {}'.format(tuple(loaded.shape), filepath)
                    )

                clip_frames.append(loaded)

            if use_explicit_frame_nbrs and missing_frame_numbers:
                raise KeyError(
                    'Explicit offline BEV chunk is incomplete for scene_token={} '
                    'anchor_frame_nbr={}. expected {} frames, loaded {}, missing frame_nbrs={}. '
                    'Please check {} and the offline BEV directory for this split.'.format(
                        scene_token,
                        frame_nbr,
                        len(frame_numbers),
                        len(clip_frames),
                        missing_frame_numbers,
                        self._offline_metadata_path(),
                    )
                )

            if len(clip_frames) == 0:
                raise KeyError(
                    'Missing offline BEV sequence for scene_token={} frame_nbr={}. '
                    'Please check {}.'.format(
                        scene_token,
                        frame_nbr,
                        self._offline_metadata_path(),
                    )
                )

            sample_bev = torch.cat(clip_frames, dim=0).unsqueeze(0)
            # Normalize each sample first so batch concat never sees
            # different temporal lengths across samples.
            sample_bev = self._uniform_temporal_length(sample_bev)
            batch_bev.append(sample_bev)

        bev_seq = torch.cat(batch_bev, dim=0).to(device=device)
        return bev_seq

    def _uniform_temporal_length(self, bev_seq):
        """Resize temporal length to expected_frames by sample/pad.

        If sequence is longer than expected_frames, frames are uniformly
        sampled. If shorter, the last frame token is repeated as padding.
        """
        # bev_seq: [B, T, HW, C]
        bsz, t, hw, c = bev_seq.shape
        if t == self.expected_frames:
            return bev_seq
        if t > self.expected_frames:
            idx = torch.linspace(0, t - 1, self.expected_frames, device=bev_seq.device)
            idx = idx.round().long().clamp(min=0, max=t - 1)
            return bev_seq.index_select(dim=1, index=idx)

        pad_len = self.expected_frames - t
        pad_token = bev_seq[:, -1:, :, :].expand(bsz, pad_len, hw, c)
        return torch.cat([bev_seq, pad_token], dim=1)

    def _extract_bev_sequence(self, img, img_metas):
        """Extract frozen BEV tokens for all frames in the clip.

        Args:
            img: [B, T, N, C, H, W]
        Returns:
            bev_seq: [B, expected_T, HW, C]
        """
        temporal_metas = self._parse_temporal_meta(img_metas)
        batch_bev = []
        for b, sample_metas in enumerate(temporal_metas):
            frame_bev = []
            frame_keys = list(sample_metas.keys())
            frame_metas = [sample_metas[k] for k in frame_keys]
            for frame_meta in frame_metas:
                if not isinstance(frame_meta, dict):
                    raise TypeError('frame_meta must be a single-frame dict.')
                if any(isinstance(k, int) for k in frame_meta.keys()):
                    raise ValueError(
                        'Temporal img_metas dict was passed into single-frame BEV extraction. '
                        'Expected per-frame metadata only.'
                    )

            frame_imgs = img[b, :len(frame_keys), ...]
            with torch.no_grad():
                # BEVFormer is frozen and used as feature extractor only.
                # Keep this branch in FP32 to avoid nearest2d BF16 errors
                # on torch builds without BF16 upsample support.
                with self._bevformer_safe_context():
                    # Run backbone/neck once for all timesteps in this sample
                    # to avoid per-frame repeated feature extraction overhead.
                    img_feats_all = self.extract_feat(img=frame_imgs.float(), img_metas=frame_metas)
                    if self.num_levels:
                        img_feats_all = img_feats_all[:self.num_levels]

                    for t_idx, frame_meta in enumerate(frame_metas):
                        per_frame_feats = [feat[t_idx:t_idx + 1, ...] for feat in img_feats_all]
                        # only_bev=True returns dense BEV tokens of shape [1, HW, C].
                        bev = self.pts_bbox_head(per_frame_feats, [frame_meta], None, only_bev=True)
                        frame_bev.append(bev)
            frame_bev = torch.stack(frame_bev, dim=1)  # [1, T, HW, C]
            batch_bev.append(frame_bev)
        bev_seq = torch.cat(batch_bev, dim=0)  # [B, T, HW, C]
        return self._uniform_temporal_length(bev_seq)

    def _encode_vision(self, bev_seq, return_intermediates=False):
        """Encode dense BEV tokens into one normalized global vision embedding.

        Args:
            bev_seq: [B, T, HW, C], typically [B, 40, 40000, 256].
        Returns:
            vision_feat: [B, 768], L2-normalized.
        """
        # bev_seq: [B, T, HW, 256]
        bsz, t, hw, c = bev_seq.shape
        if hw != self.spatial_bev_h * self.spatial_bev_w:
            raise ValueError(
                'Unexpected BEV token size: {}. spatial_bev_h * spatial_bev_w = {}.'.format(
                    hw, self.spatial_bev_h * self.spatial_bev_w)
            )

        # Broadcast 2D BEV PE to all batches and timesteps.
        spatial_pe = self.spatial_pe.view(1, 1, hw, c)
        # Apply 2D spatial PE before spatial pooling.
        bev_seq = bev_seq + spatial_pe
        tokens = bev_seq.reshape(bsz * t, hw, c)

        query = self.spatial_query.expand(tokens.shape[0], -1, -1)
        # Cross-attention style pooling: one learnable query summarizes HW tokens.
        pooled, _ = self.spatial_pool(query=query, key=tokens, value=tokens)
        pooled = pooled.squeeze(1).reshape(bsz, t, c)  # [B, T, 256]

        vision_spatial_pooled = pooled.mean(dim=1)
        vision_spatial_up = self.vision_up_proj(vision_spatial_pooled)

        pooled = self.vision_up_proj(pooled)  # [B, T, 768]
        cls_token = self.cls_token.expand(bsz, -1, -1)
        seq = torch.cat([cls_token, pooled], dim=1)  # [B, T+1, 768]
        temporal_pe = torch.cat([
            self.cls_temporal_pe,
            self.temporal_pe[:, :t, :],
        ], dim=1)
        # CLS gets dedicated PE to avoid frame index shift bugs.
        seq = seq + temporal_pe
        seq = self.temporal_encoder(seq)
        vision_temporal_cls = seq[:, 0, :]
        frame_tokens = seq[:, 1:, :]
        vision_temporal_frame_mean = frame_tokens.mean(dim=1)
        temporal_query = self.temporal_pool_query.expand(bsz, -1, -1)
        vision_temporal_attn, _ = self.temporal_pool(
            query=temporal_query,
            key=frame_tokens,
            value=frame_tokens,
        )
        vision_temporal_attn = vision_temporal_attn.squeeze(1)
        vision_netvlad = self._netvlad_pool(frame_tokens)
        if self.vision_pooling == 'spatial_mean':
            video_raw_feat = vision_spatial_up
        elif self.vision_pooling == 'temporal_attn':
            video_raw_feat = vision_temporal_attn
        elif self.vision_pooling == 'netvlad':
            video_raw_feat = vision_netvlad
        elif self.vision_pooling == 'frame_mean':
            video_raw_feat = vision_temporal_frame_mean
        else:
            video_raw_feat = vision_temporal_cls

        vision_projector_out = self.vision_projector(video_raw_feat)
        if self.use_vision_projector:
            residual_weight = self.vision_projector_residual_weight
            video_feat_pre_norm = video_raw_feat + residual_weight * (vision_projector_out - video_raw_feat)
        else:
            video_feat_pre_norm = video_raw_feat
        video_feat = F.normalize(video_feat_pre_norm, dim=-1)
        if not return_intermediates:
            return video_feat
        intermediates = {
            'vision_spatial_pooled': vision_spatial_pooled,
            'vision_spatial_up': vision_spatial_up,
            'vision_temporal_cls': vision_temporal_cls,
            'vision_temporal_frame_mean': vision_temporal_frame_mean,
            'vision_temporal_attn': vision_temporal_attn,
            'vision_netvlad': vision_netvlad,
            'vision_pre_norm': video_feat_pre_norm,
            'vision_projector_out': F.normalize(vision_projector_out, dim=-1),
            'vision_final': video_feat,
        }
        return video_feat, intermediates

    def _netvlad_pool(self, tokens):
        """Pool temporal frame tokens with trainable multi-cluster residuals."""
        assignments = F.softmax(self.vision_netvlad_assign(tokens), dim=-1)
        residuals = tokens.unsqueeze(2) - self.vision_netvlad_clusters_param.view(1, 1, -1, tokens.size(-1))
        vlad = (assignments.unsqueeze(-1) * residuals).sum(dim=1)
        vlad = F.normalize(vlad, p=2, dim=-1)
        vlad = vlad.reshape(tokens.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=-1)
        return self.vision_netvlad_proj(vlad)

    def _masked_mean_pooling(self, hidden_states, attention_mask):
        """Pool text tokens with attention mask to ignore [PAD] positions.

        Args:
            hidden_states: [B, L, C]
            attention_mask: [B, L], 1 for valid tokens and 0 for PAD.
        Returns:
            pooled: [B, C]
        """
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_sum = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return masked_sum / denom

    def _resolve_scene_tokens(self, img_metas):
        """Resolve scene_token list from temporal metadata."""
        tokens = []
        temporal_metas = self._parse_temporal_meta(img_metas)
        for sample_metas in temporal_metas:
            anchor_meta = self._get_anchor_meta(sample_metas)
            tokens.append(anchor_meta.get('scene_token', ''))
        return tokens

    def _resolve_scene_text(self, img_metas, scene_text=None, scene_tokens=None):
        """Resolve text input from explicit argument or scene metadata.

        Priority:
            1) function argument `scene_text`
            2) scene.json mapping by scene_token in img_metas
            3) raw scene_token string as fallback
        """
        if scene_text is not None:
            if isinstance(scene_text, str):
                return [scene_text]
            if isinstance(scene_text, (list, tuple)):
                return [str(x) for x in scene_text]

        if scene_tokens is None:
            scene_tokens = self._resolve_scene_tokens(img_metas)

        texts = []
        for token in scene_tokens:
            text = self.scene_token_to_text.get(token, token)
            texts.append(text)
        return texts

    def _scene_token_to_id(self, token):
        """Map scene token string to stable positive int64 id."""
        raw = hashlib.blake2b(str(token).encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(raw, byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF

    def _resolve_scene_ids(self, img_metas=None, scene_tokens=None, device=None):
        """Resolve scene ids for batch samples used by queue mask."""
        if scene_tokens is None:
            if img_metas is None:
                raise ValueError('Either img_metas or scene_tokens must be provided.')
            scene_tokens = self._resolve_scene_tokens(img_metas)

        ids = [self._scene_token_to_id(token) for token in scene_tokens]
        scene_ids = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            scene_ids = scene_ids.to(device=device)
        return scene_ids

    def _text_to_id(self, text):
        """Map text content to stable positive int64 id."""
        raw = hashlib.blake2b(str(text).encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(raw, byteorder='big', signed=False) & 0x7FFFFFFFFFFFFFFF

    def _resolve_text_ids(self, texts, device=None):
        """Resolve text ids for batch samples used by queue mask."""
        ids = [self._text_to_id(t) for t in texts]
        text_ids = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            text_ids = text_ids.to(device=device)
        return text_ids

    def _encode_text(self, texts, device, return_intermediates=False):
        """Encode text with frozen DeBERTa and trainable text projector.

        Args:
            texts: list[str], batch of scene-level descriptions.
        Returns:
            text_feat: [B, 768], L2-normalized.
        """
        batch_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt')
        batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**batch_tokens)

        text_raw_pool = self._masked_mean_pooling(outputs.last_hidden_state, batch_tokens['attention_mask'])
        if self.use_text_projector:
            text_projector_out = self.text_projector(text_raw_pool)
            text_feat = text_projector_out
        else:
            text_projector_out = self.text_projector(text_raw_pool)
            text_feat = text_raw_pool
        text_feat = F.normalize(text_feat, dim=-1)
        if not return_intermediates:
            return text_feat
        intermediates = {
            'text_raw_pool': text_raw_pool,
            'text_projector_out': F.normalize(text_projector_out, dim=-1),
            'text_final': text_feat,
        }
        return text_feat, intermediates

    def _gather_with_grad(self, x):
        """All-gather tensors for larger in-batch negatives in DDP training.

        This implementation keeps gradient for local rank by replacing gathered
        local slice with the original tensor `x`.
        """
        if not dist.is_available() or not dist.is_initialized():
            return x
        world_size = dist.get_world_size()
        if world_size <= 1:
            return x

        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)
        rank = dist.get_rank()
        gathered[rank] = x
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def _gather_no_grad(self, x):
        """All-gather detached tensors for queue update in DDP."""
        x = x.detach()
        if not dist.is_available() or not dist.is_initialized():
            return x
        world_size = dist.get_world_size()
        if world_size <= 1:
            return x

        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue_name, feats, scene_ids=None):
        """FIFO queue update with wrap-around write."""
        if feats.numel() == 0:
            return

        queue = self.text_queue if queue_name == 'text' else self.vision_queue
        queue_scene = self.text_queue_scene_id if queue_name == 'text' else self.vision_queue_scene_id
        feats = feats.to(device=queue.device, dtype=queue.dtype)
        feats = F.normalize(feats, dim=-1)

        if scene_ids is None:
            scene_ids = torch.full((feats.shape[0],), -1, dtype=torch.long, device=queue.device)
        else:
            scene_ids = scene_ids.to(device=queue.device, dtype=torch.long)

        qsize = queue.shape[0]
        if feats.shape[0] > qsize:
            feats = feats[-qsize:]
            scene_ids = scene_ids[-qsize:]
        bsz = feats.shape[0]
        ptr = int(self.queue_ptr.item())

        first = min(qsize - ptr, bsz)
        queue[ptr:ptr + first] = feats[:first]
        queue_scene[ptr:ptr + first] = scene_ids[:first]
        remain = bsz - first
        if remain > 0:
            queue[:remain] = feats[first:first + remain]
            queue_scene[:remain] = scene_ids[first:first + remain]

        ptr = (ptr + bsz) % qsize
        self.queue_ptr[0] = ptr
        self.queue_valid_len[0] = min(qsize, int(self.queue_valid_len.item()) + bsz)

    @torch.no_grad()
    def _enqueue_feature_queue(self, vision_feat, text_feat, scene_ids=None, text_ids=None, gather_ddp=False):
        """Update vision/text feature queues after each training step."""
        if not self.use_feature_queue:
            return

        if gather_ddp:
            queue_vision = self._gather_no_grad(vision_feat)
            queue_text = self._gather_no_grad(text_feat)
            if scene_ids is None:
                queue_scene_ids = None
            else:
                queue_scene_ids = self._gather_no_grad(scene_ids)
            if text_ids is None:
                queue_text_ids = None
            else:
                queue_text_ids = self._gather_no_grad(text_ids)
        else:
            queue_vision = vision_feat.detach()
            queue_text = text_feat.detach()
            if scene_ids is None:
                queue_scene_ids = None
            else:
                queue_scene_ids = scene_ids.detach()
            if text_ids is None:
                queue_text_ids = None
            else:
                queue_text_ids = text_ids.detach()

        # Enqueue vision/text in lock-step so queue_ptr/queue_valid_len are
        # advanced exactly once per training step.
        qsize = self.vision_queue.shape[0]
        queue_vision = queue_vision.to(device=self.vision_queue.device, dtype=self.vision_queue.dtype)
        queue_text = queue_text.to(device=self.text_queue.device, dtype=self.text_queue.dtype)
        queue_vision = F.normalize(queue_vision, dim=-1)
        queue_text = F.normalize(queue_text, dim=-1)

        bsz = min(queue_vision.shape[0], queue_text.shape[0])
        if bsz <= 0:
            return
        queue_vision = queue_vision[:bsz]
        queue_text = queue_text[:bsz]

        if queue_scene_ids is None:
            queue_scene_ids = torch.full((bsz,), -1, dtype=torch.long, device=self.vision_queue.device)
        else:
            queue_scene_ids = queue_scene_ids.to(device=self.vision_queue.device, dtype=torch.long)
            if queue_scene_ids.shape[0] < bsz:
                bsz = int(queue_scene_ids.shape[0])
                if bsz <= 0:
                    return
                queue_vision = queue_vision[:bsz]
                queue_text = queue_text[:bsz]
                queue_scene_ids = queue_scene_ids[:bsz]
            else:
                queue_scene_ids = queue_scene_ids[:bsz]

        if queue_text_ids is None:
            queue_text_ids = torch.full((bsz,), -1, dtype=torch.long, device=self.vision_queue.device)
        else:
            queue_text_ids = queue_text_ids.to(device=self.vision_queue.device, dtype=torch.long)
            if queue_text_ids.shape[0] < bsz:
                bsz = int(queue_text_ids.shape[0])
                if bsz <= 0:
                    return
                queue_vision = queue_vision[:bsz]
                queue_text = queue_text[:bsz]
                queue_scene_ids = queue_scene_ids[:bsz]
                queue_text_ids = queue_text_ids[:bsz]
            else:
                queue_text_ids = queue_text_ids[:bsz]

        if bsz > qsize:
            queue_vision = queue_vision[-qsize:]
            queue_text = queue_text[-qsize:]
            queue_scene_ids = queue_scene_ids[-qsize:]
            queue_text_ids = queue_text_ids[-qsize:]
            bsz = qsize

        ptr = int(self.queue_ptr.item())
        first = min(qsize - ptr, bsz)

        self.vision_queue[ptr:ptr + first] = queue_vision[:first]
        self.text_queue[ptr:ptr + first] = queue_text[:first]
        self.vision_queue_scene_id[ptr:ptr + first] = queue_scene_ids[:first]
        self.text_queue_scene_id[ptr:ptr + first] = queue_scene_ids[:first]
        self.vision_queue_text_id[ptr:ptr + first] = queue_text_ids[:first]
        self.text_queue_text_id[ptr:ptr + first] = queue_text_ids[:first]

        remain = bsz - first
        if remain > 0:
            self.vision_queue[:remain] = queue_vision[first:first + remain]
            self.text_queue[:remain] = queue_text[first:first + remain]
            self.vision_queue_scene_id[:remain] = queue_scene_ids[first:first + remain]
            self.text_queue_scene_id[:remain] = queue_scene_ids[first:first + remain]
            self.vision_queue_text_id[:remain] = queue_text_ids[first:first + remain]
            self.text_queue_text_id[:remain] = queue_text_ids[first:first + remain]

        ptr = (ptr + bsz) % qsize
        self.queue_ptr[0] = ptr
        self.queue_valid_len[0] = min(qsize, int(self.queue_valid_len.item()) + bsz)

    def _queue_ready(self):
        if not self.use_feature_queue:
            return False
        if int(self.queue_valid_len.item()) <= 0:
            return False
        return int(self.queue_step.item()) >= self.feature_queue_warmup_steps

    def _pairwise_valid_equal_mask(self, left_ids, right_ids):
        """Build pairwise equality mask and ignore unknown ids (-1)."""
        if left_ids is None or right_ids is None:
            return None
        left_ids = left_ids.view(-1)
        right_ids = right_ids.view(-1)
        eq_mask = left_ids.unsqueeze(1).eq(right_ids.unsqueeze(0))
        valid_mask = left_ids.unsqueeze(1).ge(0) & right_ids.unsqueeze(0).ge(0)
        return eq_mask & valid_mask

    def _multi_positive_infonce(self, logits, positive_mask):
        """Compute multi-positive InfoNCE from logits and positive mask."""
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            # Fallback for numerical safety; should not happen because diagonal
            # positives are always injected for gathered in-batch pairs.
            return logits.new_zeros(())

        logits = logits[valid_rows]
        positive_mask = positive_mask[valid_rows]
        neg_inf = torch.finfo(logits.dtype).min
        pos_logits = logits.masked_fill(~positive_mask, neg_inf)
        pos_lse = torch.logsumexp(pos_logits, dim=1)
        all_lse = torch.logsumexp(logits, dim=1)
        return -(pos_lse - all_lse).mean()

    def _feature_centering_loss(self, feat_mat):
        """Penalize a dominant batch-wise mean direction in feature space."""
        if feat_mat is None or feat_mat.numel() == 0:
            return self.logit_scale.new_zeros(())
        mean_feat = feat_mat.mean(dim=0)
        return mean_feat.pow(2).sum()

    def _feature_covariance_loss(self, feat_mat):
        """Penalize feature correlation and low per-dimension variance.

        The previous raw-covariance formulation was too easy to dilute after
        averaging over all off-diagonal entries in a 768-dim space. This
        variant first standardizes each dimension, then penalizes off-diagonal
        correlation and collapsed per-dimension std in the style of VICReg.
        """
        if feat_mat is None or feat_mat.dim() != 2 or feat_mat.shape[0] <= 1:
            return self.logit_scale.new_zeros(())

        centered = feat_mat - feat_mat.mean(dim=0, keepdim=True)
        var = centered.pow(2).mean(dim=0)
        std = torch.sqrt(var + 1e-4)

        # Prevent low-variance dimensions from silently collapsing even when
        # raw off-diagonal covariance is numerically tiny.
        variance_floor_loss = F.relu(1.0 - std).mean()

        standardized = centered / std.unsqueeze(0)
        corr = standardized.t().matmul(standardized) / float(feat_mat.shape[0] - 1)
        offdiag_mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
        offdiag = corr[offdiag_mask]
        if offdiag.numel() == 0:
            return variance_floor_loss
        correlation_loss = offdiag.pow(2).mean()
        return variance_floor_loss + correlation_loss

    def _vision_geometry_losses(self, vision_feat, gather_ddp=False):
        """Build auxiliary losses that spread the vision embedding geometry."""
        if self.vision_batch_centering_weight <= 0 and self.vision_covariance_reg_weight <= 0:
            return {}

        if gather_ddp:
            feat_mat = self._gather_with_grad(vision_feat)
        else:
            feat_mat = vision_feat

        losses = {}
        if self.vision_batch_centering_weight > 0:
            losses['loss_vision_centering'] = (
                self.vision_batch_centering_weight * self._feature_centering_loss(feat_mat)
            )
        if self.vision_covariance_reg_weight > 0:
            losses['loss_vision_covariance'] = (
                self.vision_covariance_reg_weight * self._feature_covariance_loss(feat_mat)
            )
        return losses

    def _contrastive_loss(self, vision_feat, text_feat, scene_ids=None, text_ids=None, gather_ddp=False, use_queue=False):
        """Compute symmetric InfoNCE loss (image->text and text->image).

        Args:
            vision_feat: [B, C], normalized.
            text_feat: [B, C], normalized.
            gather_ddp: whether to gather features across workers.
            use_queue: whether to append queue negatives.
        Returns:
            loss, i2t_top1, t2i_top1, logits_i2t
        """
        if gather_ddp:
            global_vision = self._gather_with_grad(vision_feat)
            global_text = self._gather_with_grad(text_feat)
            global_scene_ids = self._gather_no_grad(scene_ids) if scene_ids is not None else None
            global_text_ids = self._gather_no_grad(text_ids) if text_ids is not None else None
        else:
            global_vision = vision_feat
            global_text = text_feat
            global_scene_ids = scene_ids
            global_text_ids = text_ids

        local_scene_ids = None
        if scene_ids is not None:
            local_scene_ids = scene_ids.to(device=vision_feat.device, dtype=torch.long).view(-1)
        if global_scene_ids is not None:
            global_scene_ids = global_scene_ids.to(device=vision_feat.device, dtype=torch.long).view(-1)

        local_text_ids = None
        if text_ids is not None:
            local_text_ids = text_ids.to(device=vision_feat.device, dtype=torch.long).view(-1)
        if global_text_ids is not None:
            global_text_ids = global_text_ids.to(device=vision_feat.device, dtype=torch.long).view(-1)

        # Learnable temperature in log space for numerical stability.
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = scale * (vision_feat @ global_text.t())
        logits_t2i = scale * (text_feat @ global_vision.t())

        queue_scene_mask_i2t = None
        queue_scene_mask_t2i = None
        queue_text_mask_i2t = None
        queue_text_mask_t2i = None

        if use_queue and self._queue_ready():
            qlen = int(self.queue_valid_len.item())
            queue_text = self.text_queue[:qlen].detach()
            queue_vision = self.vision_queue[:qlen].detach()

            queue_logits_i2t = scale * (vision_feat @ queue_text.t())
            queue_logits_t2i = scale * (text_feat @ queue_vision.t())
            text_queue_scene_ids = self.text_queue_scene_id[:qlen].detach().to(
                device=vision_feat.device, dtype=torch.long)
            vision_queue_scene_ids = self.vision_queue_scene_id[:qlen].detach().to(
                device=vision_feat.device, dtype=torch.long)
            queue_scene_mask_i2t = self._pairwise_valid_equal_mask(local_scene_ids, text_queue_scene_ids)
            queue_scene_mask_t2i = self._pairwise_valid_equal_mask(local_scene_ids, vision_queue_scene_ids)

            if (
                not self.use_multi_positive
                and self.queue_use_scene_mask
                and queue_scene_mask_i2t is not None
                and queue_scene_mask_t2i is not None
                and local_scene_ids.shape[0] == vision_feat.shape[0]
            ):
                queue_logits_i2t = queue_logits_i2t.masked_fill(queue_scene_mask_i2t, -1e4)
                queue_logits_t2i = queue_logits_t2i.masked_fill(queue_scene_mask_t2i, -1e4)

            # In trainval, different scenes can share identical descriptions.
            # Mask same-text queue entries to avoid false negatives.
            text_queue_text_ids = self.text_queue_text_id[:qlen].detach().to(
                device=vision_feat.device, dtype=torch.long)
            vision_queue_text_ids = self.vision_queue_text_id[:qlen].detach().to(
                device=vision_feat.device, dtype=torch.long)
            queue_text_mask_i2t = self._pairwise_valid_equal_mask(local_text_ids, text_queue_text_ids)
            queue_text_mask_t2i = self._pairwise_valid_equal_mask(local_text_ids, vision_queue_text_ids)

            if (
                not self.use_multi_positive
                and queue_text_mask_i2t is not None
                and queue_text_mask_t2i is not None
                and local_text_ids.shape[0] == vision_feat.shape[0]
            ):
                queue_logits_i2t = queue_logits_i2t.masked_fill(queue_text_mask_i2t, -1e4)
                queue_logits_t2i = queue_logits_t2i.masked_fill(queue_text_mask_t2i, -1e4)

            logits_i2t = torch.cat([logits_i2t, queue_logits_i2t], dim=1)
            logits_t2i = torch.cat([logits_t2i, queue_logits_t2i], dim=1)

        if gather_ddp and dist.is_available() and dist.is_initialized():
            # Local labels map to the corresponding slice in gathered global batch.
            rank = dist.get_rank()
            labels = torch.arange(vision_feat.shape[0], device=vision_feat.device) + rank * vision_feat.shape[0]
        else:
            labels = torch.arange(vision_feat.shape[0], device=vision_feat.device)

        if self.use_multi_positive:
            # Build positive masks on global in-batch columns first.
            global_cols = global_text.shape[0]
            positive_mask_i2t = torch.zeros_like(logits_i2t, dtype=torch.bool)
            positive_mask_t2i = torch.zeros_like(logits_t2i, dtype=torch.bool)
            positive_mask_i2t.scatter_(1, labels.view(-1, 1), True)
            positive_mask_t2i.scatter_(1, labels.view(-1, 1), True)

            global_scene_mask = self._pairwise_valid_equal_mask(local_scene_ids, global_scene_ids)
            if global_scene_mask is not None and global_scene_mask.shape[1] == global_cols:
                positive_mask_i2t[:, :global_cols] |= global_scene_mask
                positive_mask_t2i[:, :global_cols] |= global_scene_mask

            global_text_mask = self._pairwise_valid_equal_mask(local_text_ids, global_text_ids)
            if global_text_mask is not None and global_text_mask.shape[1] == global_cols:
                positive_mask_i2t[:, :global_cols] |= global_text_mask
                positive_mask_t2i[:, :global_cols] |= global_text_mask

            if queue_scene_mask_i2t is not None:
                positive_mask_i2t[:, global_cols:] |= queue_scene_mask_i2t
            if queue_scene_mask_t2i is not None:
                positive_mask_t2i[:, global_cols:] |= queue_scene_mask_t2i
            if queue_text_mask_i2t is not None:
                positive_mask_i2t[:, global_cols:] |= queue_text_mask_i2t
            if queue_text_mask_t2i is not None:
                positive_mask_t2i[:, global_cols:] |= queue_text_mask_t2i

            loss_i2t = self._multi_positive_infonce(logits_i2t, positive_mask_i2t)
            loss_t2i = self._multi_positive_infonce(logits_t2i, positive_mask_t2i)
            loss = 0.5 * (loss_i2t + loss_t2i)

            with torch.no_grad():
                i2t_top1_idx = logits_i2t.argmax(dim=1, keepdim=True)
                t2i_top1_idx = logits_t2i.argmax(dim=1, keepdim=True)
                i2t_top1 = positive_mask_i2t.gather(1, i2t_top1_idx).float().mean()
                t2i_top1 = positive_mask_t2i.gather(1, t2i_top1_idx).float().mean()
        else:
            loss_i2t = F.cross_entropy(logits_i2t, labels)
            loss_t2i = F.cross_entropy(logits_t2i, labels)
            loss = 0.5 * (loss_i2t + loss_t2i)

            with torch.no_grad():
                i2t_top1 = (logits_i2t.argmax(dim=1) == labels).float().mean()
                t2i_top1 = (logits_t2i.argmax(dim=1) == labels).float().mean()

        return loss, i2t_top1, t2i_top1, logits_i2t

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      scene_text=None,
                      **kwargs):
        """Training forward for alignment task only.

        Expects temporal multi-camera image clip and scene-level text target.
        Returns MMDetection-style loss dict.
        """
        offline_mode = self.run_mode == 'offline_train'
        if img is None and not offline_mode:
            raise ValueError('img is required for BEVFormerDebertaAlign in non-offline mode.')
        if img_metas is None:
            raise ValueError('img_metas is required for BEVFormerDebertaAlign.')

        device = img.device if img is not None else self.logit_scale.device

        with self._amp_autocast_context():
            if offline_mode:
                bev_seq = self._load_offline_bev_sequence(img_metas, device=device)
            else:
                bev_seq = self._extract_bev_sequence(img, img_metas)

            if self.run_mode == 'extract':
                self._dump_bev_features(bev_seq, img_metas)
                # Keep training loop compatible while skipping optimization target.
                # Use an explicit grad-enabled scalar to avoid backward failure
                # when bev_seq is produced under no_grad in extract mode.
                return {
                    'loss_align': torch.zeros((), device=bev_seq.device, requires_grad=True),
                    'acc_i2t_top1': bev_seq.new_tensor(0.0),
                    'acc_t2i_top1': bev_seq.new_tensor(0.0),
                }

            return_intermediates = bool(getattr(self, 'return_intermediate_feats', False))
            need_vision_geometry_feat = (
                self.vision_batch_centering_weight > 0
                or self.vision_covariance_reg_weight > 0
            )
            if return_intermediates or need_vision_geometry_feat:
                vision_feat, vision_intermediates = self._encode_vision(
                    bev_seq,
                    return_intermediates=True,
                )
            else:
                vision_feat = self._encode_vision(bev_seq)
                vision_intermediates = None
            scene_tokens = self._resolve_scene_tokens(img_metas)
            scene_ids = self._resolve_scene_ids(scene_tokens=scene_tokens, device=device)
            texts = self._resolve_scene_text(
                img_metas,
                scene_text=scene_text,
                scene_tokens=scene_tokens,
            )
            text_ids = self._resolve_text_ids(texts, device=device)
            if return_intermediates:
                text_feat, text_intermediates = self._encode_text(
                    texts,
                    device=device,
                    return_intermediates=True,
                )
            else:
                text_feat = self._encode_text(texts, device=device)
                text_intermediates = None

            # Queue entries collected during warmup can be very stale when
            # warmup is long (e.g. trainval). Reset once at the transition
            # step to avoid a sudden loss/metric collapse.
            if (
                self.use_feature_queue
                and self.feature_queue_warmup_steps > 0
                and int(self.queue_step.item()) == self.feature_queue_warmup_steps
                and int(self.queue_valid_len.item()) > 0
            ):
                self.queue_ptr.zero_()
                self.queue_valid_len.zero_()
                self.text_queue_scene_id.fill_(-1)
                self.vision_queue_scene_id.fill_(-1)
                self.text_queue_text_id.fill_(-1)
                self.vision_queue_text_id.fill_(-1)

            loss, i2t_top1, t2i_top1, _ = self._contrastive_loss(
                vision_feat,
                text_feat,
                scene_ids=scene_ids,
                text_ids=text_ids,
                gather_ddp=self.gather_ddp,
                use_queue=self.use_feature_queue,
            )
            vision_geometry_feat = vision_feat
            if vision_intermediates is not None:
                vision_geometry_feat = vision_intermediates.get('vision_pre_norm', vision_feat)
            aux_losses = self._vision_geometry_losses(
                vision_geometry_feat,
                gather_ddp=self.gather_ddp,
            )

            self._enqueue_feature_queue(
                vision_feat,
                text_feat,
                scene_ids=scene_ids,
                text_ids=text_ids,
                gather_ddp=self.gather_ddp,
            )
            self.queue_step.add_(1)

        return {
            'loss_align': loss,
            'acc_i2t_top1': i2t_top1,
            'acc_t2i_top1': t2i_top1,
            **aux_losses,
        }

    def forward_test(self, img_metas, img=None, scene_text=None, **kwargs):
        """Inference forward returning retrieval-oriented diagnostics.

        Returns per-sample top-1 prediction and batch-level retrieval accuracy
        computed within current inference batch.
        """
        offline_mode = self.run_mode in {'offline_infer', 'offline_infer_validate'}
        if img is None and not offline_mode:
            raise ValueError('img is required for BEVFormerDebertaAlign in non-offline mode.')

        device = img.device if img is not None else self.logit_scale.device

        with self._amp_autocast_context():
            if offline_mode:
                bev_seq = self._load_offline_bev_sequence(img_metas, device=device)
            else:
                bev_seq = self._extract_bev_sequence(img, img_metas)

            if self.run_mode == 'extract':
                self._dump_bev_features(bev_seq, img_metas)
                return [{'extract_saved': True} for _ in range(bev_seq.shape[0])]

            return_intermediates = bool(getattr(self, 'return_intermediate_feats', False))
            if return_intermediates:
                vision_feat, vision_intermediates = self._encode_vision(
                    bev_seq,
                    return_intermediates=True,
                )
            else:
                vision_feat = self._encode_vision(bev_seq)
                vision_intermediates = None
            scene_tokens = self._resolve_scene_tokens(img_metas)
            texts = self._resolve_scene_text(
                img_metas,
                scene_text=scene_text,
                scene_tokens=scene_tokens,
            )
            temporal_metas = self._parse_temporal_meta(img_metas)
            scene_tokens = []
            for sample_metas in temporal_metas:
                scene_token, _, _ = self._get_scene_token_and_group(sample_metas)
                scene_tokens.append(scene_token)
            if return_intermediates:
                text_feat, text_intermediates = self._encode_text(
                    texts,
                    device=device,
                    return_intermediates=True,
                )
            else:
                text_feat = self._encode_text(texts, device=device)
                text_intermediates = None
            loss_align, i2t_top1, t2i_top1, logits = self._contrastive_loss(
                vision_feat,
                text_feat,
                gather_ddp=False,
                use_queue=False,
            )

        results = []
        for i in range(logits.shape[0]):
            result = {
                'i2t_top1': int(logits[i].argmax().item()),
                'i2t_score': float(logits[i].max().item()),
                'loss_align': float(loss_align.item()),
                'acc_i2t_top1': float(i2t_top1.item()),
                'acc_t2i_top1': float(t2i_top1.item()),
                # Export normalized embeddings for global retrieval evaluation
                # in validation scripts (full NxN similarity matrix).
                'vision_feat': vision_feat[i].detach().cpu(),
                'text_feat': text_feat[i].detach().cpu(),
                'scene_token': scene_tokens[i],
                'scene_text': texts[i],
            }
            if return_intermediates:
                embedding_layers = {}
                for name, value in vision_intermediates.items():
                    embedding_layers[name] = value[i].detach().cpu()
                for name, value in text_intermediates.items():
                    embedding_layers[name] = value[i].detach().cpu()
                result['embedding_layers'] = embedding_layers
            results.append(result)
        return results

    def trainable_state_dict(self):
        """Return only parameters of trainable alignment modules.

        This is used by a custom checkpoint hook to save lightweight
        alignment-only weights.
        """
        trainable = {
            k: v
            for k, v in self.state_dict().items()
            if (
                k.startswith('spatial_pe')
                or k.startswith('spatial_query')
                or k.startswith('spatial_pool')
                or k.startswith('vision_up_proj')
                or k.startswith('cls_token')
                or k.startswith('cls_temporal_pe')
                or k.startswith('temporal_pe')
                or k.startswith('temporal_encoder')
                or k.startswith('temporal_pool_query')
                or k.startswith('temporal_pool')
                or k.startswith('vision_netvlad_assign')
                or k.startswith('vision_netvlad_clusters_param')
                or k.startswith('vision_netvlad_proj')
                or k.startswith('vision_projector')
                or k.startswith('text_projector')
                or k.startswith('logit_scale')
            )
        }
        return trainable
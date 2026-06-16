import json
import math
import os
import secrets
import warnings
from collections import OrderedDict

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
                 gather_ddp=True,
                 run_mode='online',
                 offline_bev_dir='',
                 offline_bev_dir_by_split=None,
                 offline_split=None,
                 offline_metadata_name='bev_feature.json',
                 offline_dump_overwrite=False,
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
            gather_ddp (bool): Whether to all-gather features across DDP
                workers when computing contrastive loss.
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
        self.scene_json = scene_json
        self.scene_text_field = scene_text_field
        self.spatial_bev_h = spatial_bev_h
        self.spatial_bev_w = spatial_bev_w
        self.gather_ddp = gather_ddp
        self.run_mode = run_mode
        self.offline_bev_dir = offline_bev_dir
        self.offline_bev_dir_by_split = offline_bev_dir_by_split or {}
        self.offline_split = offline_split
        self.offline_metadata_name = offline_metadata_name
        self.offline_dump_overwrite = offline_dump_overwrite

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

        if not isinstance(self.offline_bev_dir_by_split, dict):
            raise TypeError('offline_bev_dir_by_split must be a dict when provided.')
        if self.offline_split is None:
            self.offline_split = self._default_offline_split()

        self._offline_records = []
        self._offline_record_index = {}
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

        self.vision_projector = nn.Sequential(
            nn.Linear(proj_dims, proj_dims),
            nn.GELU(),
            nn.Linear(proj_dims, proj_dims))

        self.text_projector = nn.Sequential(
            nn.Linear(proj_dims, proj_dims),
            nn.GELU(),
            nn.Linear(proj_dims, proj_dims))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

        self.scene_token_to_text = self._load_scene_text_map()
        self.scene_token_to_name = self._load_scene_name_map()
        self.scene_sample_order = self._load_scene_sample_order_map()
        self._freeze_visual_backbone()

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
            key = '{}::{}'.format(scene_token, int(frame_nbr))

            record = self._offline_record_index.get(key)
            if record is None:
                raise KeyError(
                    'Missing offline BEV index in metadata for scene_token={} frame_nbr={}. '
                    'Please check {}.'.format(
                        scene_token,
                        frame_nbr,
                        self._offline_metadata_path(),
                    )
                )

            filename = record.get('filename', '')
            if not filename:
                raise KeyError(
                    'Offline metadata entry missing filename for scene_token={} frame_nbr={}. '
                    'Please check {}.'.format(
                        scene_token,
                        frame_nbr,
                        self._offline_metadata_path(),
                    )
                )
            active_dir = self._get_active_offline_bev_dir()
            filepath = os.path.join(active_dir, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    'Offline BEV feature not found for {} frame {}: {}'.format(
                        scene_token, frame_nbr, filepath)
                )

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

            batch_bev.append(loaded)

        bev_seq = torch.stack(batch_bev, dim=0).to(device=device)
        return self._uniform_temporal_length(bev_seq)

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
            for t_idx, frame_key in enumerate(frame_keys):
                frame_img = img[b:b + 1, t_idx, ...]
                frame_meta = [sample_metas[frame_key]]
                if not isinstance(frame_meta[0], dict):
                    raise TypeError('frame_meta must be a single-frame dict.')
                if any(isinstance(k, int) for k in frame_meta[0].keys()):
                    raise ValueError(
                        'Temporal img_metas dict was passed into single-frame BEV extraction. '
                        'Expected per-frame metadata only.'
                    )
                with torch.no_grad():
                    # BEVFormer is frozen and used as feature extractor only.
                    img_feats = self.extract_feat(img=frame_img, img_metas=frame_meta)
                    if self.num_levels:
                        img_feats = img_feats[:self.num_levels]
                    # only_bev=True returns dense BEV tokens of shape [1, HW, C].
                    bev = self.pts_bbox_head(img_feats, frame_meta, None, only_bev=True)
                frame_bev.append(bev)
            frame_bev = torch.stack(frame_bev, dim=1)  # [1, T, HW, C]
            batch_bev.append(frame_bev)
        bev_seq = torch.cat(batch_bev, dim=0)  # [B, T, HW, C]
        return self._uniform_temporal_length(bev_seq)

    def _encode_vision(self, bev_seq):
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
        video_feat = seq[:, 0, :]
        # Additional projector for manifold alignment on vision side.
        video_feat = self.vision_projector(video_feat)
        return F.normalize(video_feat, dim=-1)

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

    def _resolve_scene_text(self, img_metas, scene_text=None):
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

        texts = []
        temporal_metas = self._parse_temporal_meta(img_metas)
        for sample_metas in temporal_metas:
            anchor_meta = self._get_anchor_meta(sample_metas)
            token = anchor_meta.get('scene_token', '')
            text = self.scene_token_to_text.get(token, token)
            texts.append(text)
        return texts

    def _encode_text(self, texts, device):
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
            return_tensors='pt')
        batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**batch_tokens)

        text_feat = self._masked_mean_pooling(outputs.last_hidden_state, batch_tokens['attention_mask'])
        text_feat = self.text_projector(text_feat)
        return F.normalize(text_feat, dim=-1)

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

    def _contrastive_loss(self, vision_feat, text_feat, gather_ddp=False):
        """Compute symmetric InfoNCE loss (image->text and text->image).

        Args:
            vision_feat: [B, C], normalized.
            text_feat: [B, C], normalized.
            gather_ddp: whether to gather features across workers.
        Returns:
            loss, i2t_top1, t2i_top1, logits_i2t
        """
        if gather_ddp:
            global_vision = self._gather_with_grad(vision_feat)
            global_text = self._gather_with_grad(text_feat)
        else:
            global_vision = vision_feat
            global_text = text_feat

        # Learnable temperature in log space for numerical stability.
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i2t = scale * (vision_feat @ global_text.t())
        logits_t2i = scale * (text_feat @ global_vision.t())

        if gather_ddp and dist.is_available() and dist.is_initialized():
            # Local labels map to the corresponding slice in gathered global batch.
            rank = dist.get_rank()
            labels = torch.arange(vision_feat.shape[0], device=vision_feat.device) + rank * vision_feat.shape[0]
        else:
            labels = torch.arange(vision_feat.shape[0], device=vision_feat.device)

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
        if img is None:
            raise ValueError('img is required for BEVFormerDebertaAlign.')
        if img_metas is None:
            raise ValueError('img_metas is required for BEVFormerDebertaAlign.')

        if self.run_mode == 'offline_train':
            bev_seq = self._load_offline_bev_sequence(img_metas, device=img.device)
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

        vision_feat = self._encode_vision(bev_seq)
        texts = self._resolve_scene_text(img_metas, scene_text=scene_text)
        text_feat = self._encode_text(texts, device=img.device)
        loss, i2t_top1, t2i_top1, _ = self._contrastive_loss(
            vision_feat,
            text_feat,
            gather_ddp=self.gather_ddp,
        )

        return {
            'loss_align': loss,
            'acc_i2t_top1': i2t_top1,
            'acc_t2i_top1': t2i_top1,
        }

    def forward_test(self, img_metas, img=None, scene_text=None, **kwargs):
        """Inference forward returning retrieval-oriented diagnostics.

        Returns per-sample top-1 prediction and batch-level retrieval accuracy
        computed within current inference batch.
        """
        if img is None:
            raise ValueError('img is required for BEVFormerDebertaAlign.')

        if self.run_mode in {'offline_infer', 'offline_infer_validate'}:
            bev_seq = self._load_offline_bev_sequence(img_metas, device=img.device)
        else:
            bev_seq = self._extract_bev_sequence(img, img_metas)

        if self.run_mode == 'extract':
            self._dump_bev_features(bev_seq, img_metas)
            return [{'extract_saved': True} for _ in range(bev_seq.shape[0])]

        vision_feat = self._encode_vision(bev_seq)
        texts = self._resolve_scene_text(img_metas, scene_text=scene_text)
        text_feat = self._encode_text(texts, device=img.device)
        _, i2t_top1, t2i_top1, logits = self._contrastive_loss(
            vision_feat,
            text_feat,
            gather_ddp=False,
        )

        results = []
        for i in range(logits.shape[0]):
            results.append({
                'i2t_top1': int(logits[i].argmax().item()),
                'i2t_score': float(logits[i].max().item()),
                'acc_i2t_top1': float(i2t_top1.item()),
                'acc_t2i_top1': float(t2i_top1.item()),
                'scene_text': texts[i],
            })
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
                or k.startswith('vision_projector')
                or k.startswith('text_projector')
                or k.startswith('logit_scale')
            )
        }
        return trainable
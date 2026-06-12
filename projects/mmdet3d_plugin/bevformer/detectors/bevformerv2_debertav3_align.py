import json
import math
import os
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
                 **kwargs):
        """Initialize BEVFormer-DeBERTa alignment model.

        Args:
            text_model_name (str): Hugging Face model name/path for the frozen
                text encoder (e.g. microsoft/deberta-v3-base).
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

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
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
            first_key = next(iter(sample_metas.keys()))
            token = sample_metas[first_key].get('scene_token', '')
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

        bev_seq = self._extract_bev_sequence(img, img_metas)
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

        bev_seq = self._extract_bev_sequence(img, img_metas)
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
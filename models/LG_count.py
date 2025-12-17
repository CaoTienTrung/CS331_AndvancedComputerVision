# models/LG_count.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from models.models_crossvit import CrossAttentionBlock, ConvCrossAttentionBlock
from util.pos_embed import positional_encoding_1d
import clip
from torchvision import transforms
import einops
import functools
import operator


# =========================
# Stage1 (RichCount Stage1)
# =========================
class ImageFFN(nn.Module):
    """FFN on top of CLIP image CLS embedding (D=512 for ViT-B)."""
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextAdapter(nn.Module):
    """Adapter for CLIP text embedding (D=512 for ViT-B)."""
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.bn(h)
        h = self.dropout(h)
        return self.fc3(h)


@dataclass
class Stage1Config:
    clip_name: str = "ViT-B/16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_ckpt_path: str = ""          # empty => disable
    use_stage1_ffn: bool = True
    use_stage1_adapter: bool = True
    force_eval: bool = True             # BN inside => recommended


class Stage1Modules(nn.Module):
    """
    Holds FFN + Adapter, optionally loads from a stage1 checkpoint.
    We DO NOT need a second CLIP instance; we only load weights for ffn/adapter.
    """
    def __init__(self, embed_dim: int, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg
        self.ffn = ImageFFN(embed_dim, embed_dim)
        self.adapter = TextAdapter(embed_dim, embed_dim)
        self.to(cfg.device)

    @staticmethod
    def _strip_prefix(k: str) -> str:
        # best-effort strip common prefixes
        for p in ["model.", "stage1.", "module."]:
            if k.startswith(p):
                return k[len(p):]
        return k

    def load_stage1_checkpoint(self, ckpt_path: str, strict: bool = False):
        ckpt = torch.load(ckpt_path, map_location=self.cfg.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError(f"Unrecognized ckpt type: {type(ckpt)}")

        mapped = {}
        for k, v in sd.items():
            k2 = self._strip_prefix(k)

            # direct
            if k2.startswith("ffn."):
                mapped[k2] = v
                continue
            if k2.startswith("adapter."):
                mapped[k2] = v
                continue

            # heuristic routing
            if "ffn" in k2:
                kk = k2.replace("img_ffn", "ffn").replace("image_ffn", "ffn")
                if kk.startswith("ffn."):
                    mapped[kk] = v
            if "adapter" in k2:
                kk = k2.replace("text_adapter", "adapter")
                if kk.startswith("adapter."):
                    mapped[kk] = v

        incompat = self.load_state_dict(mapped, strict=strict)
        missing, unexpected = incompat.missing_keys, incompat.unexpected_keys
        print(f"[Stage1] Loaded: {ckpt_path}")
        print(f"[Stage1] Missing={len(missing)} Unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[Stage1] Missing keys sample:", missing[:10])
        if len(unexpected) > 0:
            print("[Stage1] Unexpected keys sample:", unexpected[:10])

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def force_eval_mode(self):
        self.eval()
        self.ffn.eval()
        self.adapter.eval()


# =========================
# Main LGCount models
# =========================
class LGCount(nn.Module):
    def __init__(self,
                 fim_depth: int = 4,
                 fim_num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 norm_layer=nn.LayerNorm,
                 use_vpt: bool = True,
                 vpt_width: int = 2,
                 vpt_depth: int = 2,
                 use_coop: bool = True,
                 coop_width: int = 2,
                 backbone: str = "b16",
                 use_fim: bool = True,
                 use_mixed_fim: bool = False,
                 unfreeze_vit: bool = False,
                 contrast_pre_epoch: int = 20,
                 # ---- Stage1 injection ----
                 stage1_ckpt_path: str = "",
                 use_stage1_ffn: bool = True,
                 use_stage1_adapter: bool = True,
                 stage1_device: Optional[str] = None,
                 stage1_force_eval: bool = True):
        super().__init__()

        # --------------------------------------------------------------------------
        # CLIP backbone
        if backbone == "b16":
            self.clip, _ = clip.load("ViT-B/16")
            self.n_patches = 14 * 14
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "b32":
            self.clip, _ = clip.load("ViT-B/32")
            self.n_patches = 7 * 7
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "l14":
            self.clip, _ = clip.load("ViT-L/14")
            self.n_patches = 16 * 16
            self.clip_hidden_dim = 1024
            self.clip_out_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.clip = self.clip.to('cuda')
        if unfreeze_vit:
            self.clip = self.clip.float()
        self.clip.requires_grad_(False)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # --------------------------------------------------------------------------
        # Stage1 modules (FFN + Adapter) injection
        self.stage1 = None
        self.use_stage1_ffn = bool(use_stage1_ffn)
        self.use_stage1_adapter = bool(use_stage1_adapter)

        if stage1_device is None:
            stage1_device = "cuda" if torch.cuda.is_available() else "cpu"

        if stage1_ckpt_path:
            cfg = Stage1Config(
                device=stage1_device,
                stage1_ckpt_path=stage1_ckpt_path,
                use_stage1_ffn=self.use_stage1_ffn,
                use_stage1_adapter=self.use_stage1_adapter,
                force_eval=stage1_force_eval
            )
            self.stage1 = Stage1Modules(self.clip_out_dim, cfg)
            self.stage1.load_stage1_checkpoint(stage1_ckpt_path, strict=False)
            self.stage1.freeze_all()
            if cfg.force_eval:
                self.stage1.force_eval_mode()
        else:
            # No ckpt: still allow "disabled" cleanly
            self.stage1 = None

        # --------------------------------------------------------------------------
        # Prompt tuning modules
        self.use_vpt = use_vpt
        self.use_coop = use_coop
        self.vpt_width = vpt_width if use_vpt else 0
        self.vpt_depth = vpt_depth if use_vpt else 0
        self.coop_width = coop_width if use_coop else 0

        self.img_encoder = CLIPViT(
            self.clip,
            self.clip_hidden_dim,
            use_vpt=self.use_vpt,
            vpt_width=self.vpt_width,
            vpt_depth=self.vpt_depth,
            unfreeze=unfreeze_vit,
            contrast_pre_epoch=contrast_pre_epoch,
            stage1_ffn=(self.stage1.ffn if self.stage1 is not None else None),
            use_stage1_ffn=self.use_stage1_ffn
        )

        self.text_encoder = CLIPTextTransformer(
            self.clip,
            use_coop=self.use_coop,
            n_ctx=self.coop_width,
            contrast_pre_epoch=contrast_pre_epoch,
            stage1_adapter=(self.stage1.adapter if self.stage1 is not None else None),
            use_stage1_adapter=self.use_stage1_adapter
        )

        # --------------------------------------------------------------------------
        # Contrastive Learning related
        self.patch_feat_proj = nn.Linear(self.clip_hidden_dim, self.clip_out_dim, bias=True)
        self.patch_feat_proj_contrast = nn.Linear(self.clip_hidden_dim, self.clip_out_dim, bias=True)
        nn.init.xavier_normal_(self.patch_feat_proj.weight)

        n_token = self.n_patches
        self.patch_emb_pos_embed = nn.Parameter(torch.zeros(1, n_token, self.clip_out_dim), requires_grad=False)
        decoder_pos_embed = positional_encoding_1d(self.clip_out_dim, n_token)
        self.patch_emb_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # --------------------------------------------------------------------------
        # Patch-text interaction module
        self.decoder_ln_pre = norm_layer(self.clip_out_dim)

        self.use_fim = use_fim
        self.use_mixed_fim = use_mixed_fim
        assert (not use_fim) or (not use_mixed_fim), \
            "You can not use hierarchical transformer and plain transformer at the same time!"

        self.fim_blocks = None
        if use_mixed_fim:
            self.fim_blocks = nn.ModuleList([
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                        norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution=1.),
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                        norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution=2.),
            ])
        elif use_fim:
            self.fim_blocks = nn.ModuleList([
                CrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                    norm_layer=norm_layer, drop=0.1, drop_path=0.1)
                for _ in range(fim_depth)
            ])

        self.decoder_norm = norm_layer(self.clip_out_dim)

        # --------------------------------------------------------------------------
        # CNN-based density decoder
        self.density_decoder = DensityDecoder(self.clip_out_dim, 384, use_hiearachy=use_mixed_fim)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep Stage1 in eval if requested (BN + dropout safety)
        if self.stage1 is not None and self.stage1.cfg.force_eval:
            self.stage1.force_eval_mode()
        return self

    def forward_visual_encoder(self, x, text_embedding=None):
        x = self.preprocess(x)
        _, cls_token, x = self.img_encoder(x, text_embedding)
        return cls_token, x

    def forward_decoder(self, cls_token, img_feat_patches, gt_rank_text_embedding):
        patch_feat = img_feat_patches[:, 1:, :]  # remove CLS from token sequence
        patch_embedding = self.patch_feat_proj(patch_feat)
        patch_embedding_contrast = self.patch_feat_proj_contrast(patch_feat)

        x = patch_embedding + self.patch_emb_pos_embed  # [B, N, D]
        y_ = gt_rank_text_embedding  # [B, 1, D] (or [B, K, D] in some modes)

        if self.use_mixed_fim:
            xs = []
            for blk in self.fim_blocks:
                x = blk(x, y_, cls_token)
                xs.append(self.seq_2_2d(x))
        elif self.use_fim:
            for blk in self.fim_blocks:
                x = blk(x, y_)
        else:
            x = x + y_

        x = self.decoder_norm(x)
        x2d = self.seq_2_2d(x)

        if self.use_mixed_fim:
            pred_density = self.density_decoder.forward_hierarchical(xs)
        else:
            pred_density = self.density_decoder(x2d)

        return pred_density, patch_embedding_contrast

    def forward(self, imgs, class_text, current_epoch, top_fine_text_embedding,
                return_extra: bool = False, coop_require_grad: bool = False):
        extra_out = {}

        class_text = list(class_text)
        class_text_token = clip.tokenize(class_text).to(imgs.device)

        if coop_require_grad:
            class_text_embedding = self.text_encoder(class_text_token).float()
        else:
            with torch.no_grad():
                class_text_embedding = self.text_encoder(class_text_token).float()

        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, text_embedding=None)

        pred_density, patch_embedding_contrast = self.forward_decoder(
            cls_token, img_feat_patches, top_fine_text_embedding
        )

        extra_out['class_text_embedding'] = class_text_embedding
        extra_out['patch_embedding_contrast'] = patch_embedding_contrast

        return pred_density, extra_out

    def seq_2_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    # ---- If you still use these helpers, now they are consistent ----
    def test_forward(self, imgs, rank_text_token, coop_require_grad=False):
        if coop_require_grad:
            rank_text_embedding = self.text_encoder(rank_text_token).float()
        else:
            with torch.no_grad():
                rank_text_embedding = self.text_encoder(rank_text_token).float()

        rank_text_embedding = rank_text_embedding.reshape(imgs.shape[0], -1, rank_text_embedding.shape[-1])
        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, text_embedding=None)

        # choose first candidate by default (or implement your own selection)
        gt_rank_text_embedding = rank_text_embedding[:, :1, :]
        pred_density, _ = self.forward_decoder(cls_token, img_feat_patches, gt_rank_text_embedding)
        return pred_density, {}

    def validation_forward(self, imgs, rank_text_list, is_fine=False):
        rank_text_token = clip.tokenize(rank_text_list).to(imgs.device)
        with torch.no_grad():
            rank_text_embedding = self.text_encoder(rank_text_token).float()

        rank_text_embedding = rank_text_embedding.reshape(imgs.shape[0], -1, rank_text_embedding.shape[-1])
        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, text_embedding=None)

        sim_map = F.cosine_similarity(cls_token, rank_text_embedding, dim=-1)
        _, top_indices = torch.topk(sim_map, k=1, dim=1)

        pred_density = None
        if is_fine:
            gt_rank_text_embedding = rank_text_embedding[
                torch.arange(rank_text_embedding.size(0)).unsqueeze(1), top_indices
            ]
            pred_density, _ = self.forward_decoder(cls_token, img_feat_patches, gt_rank_text_embedding)

        return top_indices, pred_density

    def train_forward(self, imgs, rank_text_list, index_GT=None, is_fine=False):
        rank_text_token = clip.tokenize(rank_text_list).to(imgs.device)
        rank_text_embedding = self.text_encoder(rank_text_token).float()

        rank_text_embedding = rank_text_embedding.reshape(imgs.shape[0], -1, rank_text_embedding.shape[-1])
        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, text_embedding=None)

        sim_map = F.cosine_similarity(cls_token, rank_text_embedding, dim=-1)
        _, top_indices = torch.topk(sim_map, k=1, dim=1)

        pred_density = None
        if is_fine:
            gt_rank_text_embedding = rank_text_embedding[
                torch.arange(rank_text_embedding.size(0)).unsqueeze(1), top_indices
            ]
            pred_density, _ = self.forward_decoder(cls_token, img_feat_patches, gt_rank_text_embedding)

        return top_indices, sim_map, pred_density


class LGCountAlign(nn.Module):
    def __init__(self,
                 fim_depth: int = 4,
                 fim_num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 norm_layer=nn.LayerNorm,
                 use_vpt: bool = True,
                 vpt_width: int = 2,
                 vpt_depth: int = 2,
                 use_coop: bool = True,
                 coop_width: int = 2,
                 backbone: str = "b16",
                 use_fim: bool = True,
                 use_mixed_fim: bool = False,
                 unfreeze_vit: bool = False,
                 contrast_pre_epoch: int = 20,
                 # ---- Stage1 injection ----
                 stage1_ckpt_path: str = "",
                 use_stage1_ffn: bool = True,
                 use_stage1_adapter: bool = True,
                 stage1_device: Optional[str] = None,
                 stage1_force_eval: bool = True):
        super().__init__()

        if backbone == "b16":
            self.clip, _ = clip.load("ViT-B/16")
            self.n_patches = 14 * 14
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "b32":
            self.clip, _ = clip.load("ViT-B/32")
            self.n_patches = 7 * 7
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "l14":
            self.clip, _ = clip.load("ViT-L/14")
            self.n_patches = 16 * 16
            self.clip_hidden_dim = 1024
            self.clip_out_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.clip = self.clip.to('cuda')
        if unfreeze_vit:
            self.clip = self.clip.float()
        self.clip.requires_grad_(False)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # Stage1 injection
        self.stage1 = None
        self.use_stage1_ffn = bool(use_stage1_ffn)
        self.use_stage1_adapter = bool(use_stage1_adapter)

        if stage1_device is None:
            stage1_device = "cuda" if torch.cuda.is_available() else "cpu"

        if stage1_ckpt_path:
            cfg = Stage1Config(
                device=stage1_device,
                stage1_ckpt_path=stage1_ckpt_path,
                use_stage1_ffn=self.use_stage1_ffn,
                use_stage1_adapter=self.use_stage1_adapter,
                force_eval=stage1_force_eval
            )
            self.stage1 = Stage1Modules(self.clip_out_dim, cfg)
            self.stage1.load_stage1_checkpoint(stage1_ckpt_path, strict=False)
            self.stage1.freeze_all()
            if cfg.force_eval:
                self.stage1.force_eval_mode()

        self.use_vpt = use_vpt
        self.use_coop = use_coop
        self.vpt_width = vpt_width if use_vpt else 0
        self.vpt_depth = vpt_depth if use_vpt else 0
        self.coop_width = coop_width if use_coop else 0

        self.img_encoder = CLIPViT(
            self.clip,
            self.clip_hidden_dim,
            use_vpt=self.use_vpt,
            vpt_width=self.vpt_width,
            vpt_depth=self.vpt_depth,
            unfreeze=unfreeze_vit,
            contrast_pre_epoch=contrast_pre_epoch,
            stage1_ffn=(self.stage1.ffn if self.stage1 is not None else None),
            use_stage1_ffn=self.use_stage1_ffn
        )

        self.text_encoder = CLIPTextTransformer(
            self.clip,
            use_coop=self.use_coop,
            n_ctx=self.coop_width,
            contrast_pre_epoch=contrast_pre_epoch,
            stage1_adapter=(self.stage1.adapter if self.stage1 is not None else None),
            use_stage1_adapter=self.use_stage1_adapter
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.stage1 is not None and self.stage1.cfg.force_eval:
            self.stage1.force_eval_mode()
        return self

    def forward_visual_encoder(self, x, text_embedding=None):
        x = self.preprocess(x)
        _, cls_token, x = self.img_encoder(x, text_embedding)
        return cls_token, x

    def forward(self, imgs, coarse_text_list, fine_text_list):
        coarse_text_list = list(np.array(coarse_text_list).flatten('F'))  # [B*12]

        with torch.no_grad():
            cls_token, _ = self.forward_visual_encoder(imgs, text_embedding=None)

            coarse_text_token = clip.tokenize(coarse_text_list).to(imgs.device)
            coarse_text_embedding = self.text_encoder(coarse_text_token).float()
            coarse_text_embedding = coarse_text_embedding.reshape(imgs.shape[0], -1, coarse_text_embedding.shape[-1])
            coarse_sim_map = F.cosine_similarity(cls_token, coarse_text_embedding, dim=-1)
            _, pred_coarse_top_indices = torch.topk(coarse_sim_map, k=1, dim=1)

            fine_list = []
            for i in range(len(pred_coarse_top_indices)):
                l = list(np.array(fine_text_list[pred_coarse_top_indices[i]]).flatten('F'))
                fine_list.append(l[5 * i:5 * i + 5])

            fine_list = list(np.array(fine_list).flatten())
            fine_text_token = clip.tokenize(fine_list).to(imgs.device)
            fine_text_embedding = self.text_encoder(fine_text_token).float()
            fine_text_embedding = fine_text_embedding.reshape(imgs.shape[0], -1, fine_text_embedding.shape[-1])
            fine_sim_map = F.cosine_similarity(cls_token, fine_text_embedding, dim=-1)
            _, pred_fine_top_indices = torch.topk(fine_sim_map, k=1, dim=1)

        top_fine_text_embedding = fine_text_embedding[
            torch.arange(fine_text_embedding.size(0)).unsqueeze(1), pred_fine_top_indices
        ]  # [B, 1, D]
        return pred_coarse_top_indices, pred_fine_top_indices, top_fine_text_embedding


class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    PATCH:
      - apply stage1_ffn on CLS embedding (after proj)
    """
    def __init__(self,
                 clip_model,
                 clip_embed_dim: int,
                 use_vpt: bool,
                 vpt_width: int,
                 vpt_depth: int = 8,
                 unfreeze: bool = False,
                 contrast_pre_epoch: int = 20,
                 stage1_ffn: Optional[nn.Module] = None,
                 use_stage1_ffn: bool = True):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.vit = clip_model.visual

        if unfreeze:
            for param in self.vit.parameters():
                param.requires_grad = True

        self.use_vpt = use_vpt
        self.vpt_depth = vpt_depth
        self.vpt_width = vpt_width
        self.epoch = 0
        self.contrast_pre_epoch = contrast_pre_epoch

        self.stage1_ffn = stage1_ffn
        self.use_stage1_ffn = use_stage1_ffn

        self.visual_prompt = None
        self.vpt_dropout = None
        self.vpt_norm = None
        self.vpt_proj = None

        if use_vpt:
            self.vpt_dropout = nn.Dropout(0.1)
            self.vpt_norm = nn.LayerNorm(clip_embed_dim, eps=1e-6)
            self.vpt_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
            nn.init.kaiming_normal_(self.vpt_proj.weight, a=0, mode='fan_out')

            patch_size = self.vit.conv1.kernel_size
            val = math.sqrt(6. / float(3 * functools.reduce(operator.mul, patch_size, 1) + self.clip_embed_dim))
            vpt = torch.empty((vpt_depth, vpt_width, clip_embed_dim))
            nn.init.uniform_(vpt, -val, val)
            self.visual_prompt = nn.Parameter(vpt)

    def forward(self, image, text_embedding=None):
        x = self.vit.conv1(image)  # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        img_patches = x

        x = torch.cat([
            self.vit.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        x = x + self.vit.positional_embedding.to(x.dtype)

        if self.use_vpt:
            vpts = einops.repeat(self.visual_prompt[0, ...], 'n d -> b n d', b=x.shape[0])
            x = torch.cat([x[:, :1, :], self.vpt_dropout(self.vpt_proj(vpts)), x[:, 1:, :]], dim=1)

        x = self.vit.ln_pre(x)
        x = x.permute(1, 0, 2)

        if (not self.use_vpt) or self.vpt_depth == 1:
            x = self.vit.transformer(x)
        else:
            x = self.deep_vpt_forward(x, text_embedding)

        x = x.permute(1, 0, 2)

        x_cls = x[:, :1, :]
        x_cls = self.vit.ln_post(x_cls)
        x_cls = x_cls @ self.vit.proj   # [B,1,512] for ViT-B

        # ---- Stage1 FFN hook ----
        if self.stage1_ffn is not None and self.use_stage1_ffn:
            B, _, D = x_cls.shape
            x2 = x_cls.reshape(B, D)
            x2 = self.stage1_ffn(x2)
            x_cls = x2.reshape(B, 1, D)

        return img_patches, x_cls, x

    def deep_vpt_forward(self, embedding_output, text_embdding=None, out_last=False):
        B = embedding_output.shape[1]
        transformer = self.vit.transformer
        assert self.vpt_depth < transformer.layers

        for i in range(transformer.layers):
            if i == 0:
                hidden_states = transformer.resblocks[i](embedding_output)
            elif i < self.vpt_depth:
                deep_prompt_emb = self.vpt_dropout(
                    self.vpt_proj(self.visual_prompt[i - 1, ...]).expand(B, -1, -1)
                ).permute(1, 0, 2)

                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1 + self.vpt_width):, :, :]
                ), dim=0)
                hidden_states = transformer.resblocks[i](hidden_states)
            elif i == self.vpt_depth:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[(1 + self.vpt_width):, :, :]
                ), dim=0)
                hidden_states = transformer.resblocks[i](hidden_states)
            else:
                hidden_states = transformer.resblocks[i](hidden_states)

            if i == (transformer.layers - 1):
                before_last_feats = self.vpt_norm(hidden_states)

        encoded = self.vpt_norm(hidden_states)
        return (before_last_feats, encoded) if out_last else encoded


class CLIPTextTransformer(nn.Module):
    """
    Transformer encoder (text) for CLIP
    PATCH:
      - apply stage1_adapter on final text embedding
    """
    def __init__(self,
                 clip_model,
                 use_coop: bool,
                 n_ctx: int = 2,
                 contrast_pre_epoch: int = 20,
                 stage1_adapter: Optional[nn.Module] = None,
                 use_stage1_adapter: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.learnable_context = None
        self.use_coop = use_coop
        self.epoch = 0
        self.contrast_pre_epoch = contrast_pre_epoch

        self.stage1_adapter = stage1_adapter
        self.use_stage1_adapter = use_stage1_adapter

        if use_coop:
            self.n_ctx = n_ctx
            context_vectors = torch.empty(self.n_ctx, self.clip_model.ln_final.weight.shape[0])
            torch.nn.init.normal_(context_vectors, std=.02)
            self.learnable_context = nn.Parameter(context_vectors)

    def forward(self, text):
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)

        if self.use_coop:
            sos_token = x[:, 0, :].unsqueeze(1)
            suffix_tokens = x[:, 1:-self.n_ctx, :]
            ctx = einops.repeat(self.learnable_context, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([sos_token, ctx, suffix_tokens], dim=1)

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.visual.conv1.weight.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        x = x.unsqueeze(1)  # [B,1,D]

        # ---- Stage1 Adapter hook ----
        if self.stage1_adapter is not None and self.use_stage1_adapter:
            B, _, D = x.shape
            x2 = x.reshape(B, D)
            x2 = self.stage1_adapter(x2)
            x = x2.reshape(B, 1, D)

        return x


class DensityDecoder(nn.Module):
    def __init__(self, in_dim: int, target_hw: int, use_hiearachy: bool = False):
        super().__init__()
        self.n_levels = 4 if use_hiearachy else 2
        self.target_hw = [target_hw, target_hw]

        convs = []
        crt_dim = in_dim
        for _ in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(crt_dim, crt_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, crt_dim // 2),
                nn.GELU()
            )
            convs.append(decode_head)
            crt_dim //= 2

        self.convs = nn.ModuleList(convs)
        self.final_conv = nn.Sequential(nn.Conv2d(crt_dim, 1, kernel_size=1, stride=1))

        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

        self.pyradim_conv = None
        if use_hiearachy:
            self.pyradim_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.GroupNorm(8, 256),
                nn.GELU()
            )

    def forward(self, x):
        for i in range(self.n_levels):
            x = self.convs[i](x)
            if i < self.n_levels - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size=self.target_hw, mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

    def forward_hierarchical(self, xs):
        x0, x1 = xs[0], xs[1]
        x = x0
        for i in range(self.n_levels):
            if i == 1 and self.pyradim_conv is not None:
                x = x + self.pyradim_conv(x1)

            x = self.convs[i](x)
            if i < self.n_levels - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

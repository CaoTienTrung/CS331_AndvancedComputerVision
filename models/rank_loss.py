import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(RankLoss, self).__init__()
        self.temperature = temperature

    def forward(self, patch_embedding, class_text_embedding, gt_density):
        gt_density = gt_density.unsqueeze(1)  # (B,1,H,W)
        gt_density = F.interpolate(gt_density, size=(224, 224), mode='nearest')
        density_mask = F.max_pool2d(gt_density, kernel_size=16, stride=16, padding=0)  # (B,1,24,24)
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3, 1)  # (B,14,14,1)

        class_text_embedding = class_text_embedding.unsqueeze(1).expand(-1, 14, 14, -1)

        fused_text_embedding_map = class_text_embedding
        pos_mask = density_mask.squeeze(-1)  # (B,14,14)

        patch_embeddings = patch_embedding.reshape(-1, 14, 14, 512)
        sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map, dim=-1)  # (B,14,14)
        sim_map = torch.exp(sim_map / self.temperature)

        rank_loss_list = []
        for i in range(sim_map.shape[0]):
            positive_scores = sim_map[i][pos_mask[i]]
            negative_scores = sim_map[i][~pos_mask[i]]

            if positive_scores.numel() == 0 or negative_scores.numel() == 0:
                continue

            pos_neg_diff = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
            rank_loss = torch.clamp(2.0 - pos_neg_diff, min=0).mean()
            rank_loss_list.append(rank_loss)

        # === phần fix quan trọng ===
        if len(rank_loss_list) == 0:
            # không có ảnh nào có cả pos & neg → rank loss = 0
            return torch.tensor(
                0.0,
                device=sim_map.device,
                dtype=sim_map.dtype,
            )

        rank_loss_stack = torch.stack(rank_loss_list)
        loss = rank_loss_stack.mean()
        return loss
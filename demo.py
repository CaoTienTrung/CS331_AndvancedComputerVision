from datasets import *
import torch
from tqdm import tqdm
from models import *
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import warnings
import logging
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, default_collate
from run import SCALE_FACTOR, Engine
import gradio as gr
import cv2
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# ===============================
# Sliding window utilities
# ===============================
def window_composite(patches, window_size=(384, 384), stride=128):
    image = None
    patch_h, patch_w = window_size
    for i, patch in enumerate(patches):
        if i == 0:
            image = patch
        else:
            blend_width = patch_w - stride
            prev_to_blend = image[:, :, -blend_width:]
            next_to_blend = patch[:, :, :blend_width]
            blend_factor = torch.sigmoid(
                torch.tensor(np.linspace(-3, 3, blend_width), device=image.device)
            )
            blend = (1 - blend_factor) * prev_to_blend + blend_factor * next_to_blend
            image[:, :, -blend_width:] = blend
            patch_remain = patch[:, :, blend_width:]
            image = torch.cat([image, patch_remain], dim=-1)
    return image


def sliding_window(image, window_size=(384, 384), stride=128):
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    image = np.pad(
        image,
        ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)),
        'constant'
    )
    h, w, _ = image.shape
    assert h == 384, "FSC-147 assume image height is 384."

    patches, intervals = [], []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])

    patches = np.array(patches).transpose(0, 3, 1, 2)
    return patches, np.array(intervals)


# ===============================
# Load model
# ===============================
args = get_parser()
config = load_config(args.config)
logg(config['training']['log_file'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine = Engine(config)
engine.reload()

model = engine.model.to(device)
model.eval()


# ===============================
# Gradio inference
# ===============================
def infer(img, prompt):
    """
    img: np.ndarray (H, W, 3), RGB, uint8
    prompt: str
    """
    model.eval()

    with torch.no_grad():
        # ---- save input for GroundingDINO / exampler ----
        cv2.imwrite("input.jpg", img[..., ::-1])  # RGB -> BGR

        # ---- preprocess image for model ----
        image = Image.open("input.jpg").convert("RGB")
        W, H = image.size
        new_H = 16 * (H // 16)
        new_W = 16 * (W // 16)

        image = transforms.Resize((new_H, new_W))(image)
        image = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # ---- get exampler crops ----
        img_src, img_gd = load_image("input.jpg")
        img_src = [img_src]
        img_gd = [img_gd]
        cls_name_list = [prompt]

        examplers = engine.get_exampler.get_highest_score_crop(
            img_gd,
            img_src,
            cls_name_list,
            box_threshold=0.2,  
            keep_area=KEEP_AREA,
            device=device
        )
        if examplers is not None:
            examplers = examplers.to(device)

        # ---- forward model ----
        output, _ = model(
            image,
            cls_name_list,
            coop_require_grad=config['training'].get('coop_training', False),
            examplers=examplers
        )

        # ---- count ----
        pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()

        # ---- density map ----
        pred_density = output[0].detach().cpu().numpy()
        pred_density = pred_density / (pred_density.max() + 1e-6)

        pred_density_vis = 1.0 - pred_density
        pred_density_vis = cv2.applyColorMap(
            np.uint8(255 * pred_density_vis),
            cv2.COLORMAP_JET
        )
        pred_density_vis = pred_density_vis.astype(np.float32) / 255.0
        pred_density_vis = pred_density_vis[..., ::-1]  # BGR -> RGB

        # ---- prepare Gradio image ----
        if img.dtype != np.float32:
            img_vis = img.astype(np.float32) / 255.0
        else:
            img_vis = img.copy()
            if img_vis.max() > 1.0:
                img_vis /= 255.0

        # ---- overlay ----
        img_vis = cv2.resize(img_vis, (pred_density_vis.shape[1], pred_density_vis.shape[0]))
        heatmap_pred = 0.33 * img_vis + 0.67 * pred_density_vis
        heatmap_pred = heatmap_pred / (heatmap_pred.max() + 1e-6)

    return heatmap_pred, pred_cnt


# ===============================
# Gradio UI
# ===============================
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(label="Image"),
        gr.Textbox(lines=1, label="Prompt (What would you like to count)")
    ],
    outputs=["image", "number"],
    title="DEMO ",
    description="A unified counting model with exemplar guidance."
)

demo.launch(share=True)

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


args = get_parser()
config = load_config(args.config)
logg(config['training']['log_file'])

path_to_images = 'Dataset/images_384_VarV2'
path_to_json = 'Dataset/FSC-147-S.json'

with open(path_to_json, 'r') as f:
    data = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine = Engine(config)
engine.reload()
model = engine.model.to(device)
model.eval()

total_abs_err = 0.0
total_sq_err  = 0.0
total_samples = 0

for img_file in data.keys():
    gt = data[img_file]['count']
    cls_name = data[img_file]['class']
    img_path = os.path.join(path_to_images, img_file)

    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    new_H = 16 * (H // 16)
    new_W = 16 * (W // 16)
    image = transforms.Resize((new_H, new_W))(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    img_src, img_gd = load_image(img_path)
    img_src = [img_src]
    img_gd  = [img_gd]
    cls_name_list = [cls_name]

    examplers = engine.get_exampler.get_highest_score_crop(
        img_gd, img_src, cls_name_list,
        box_threshold=BOX_THRESHOLD,
        keep_area=KEEP_AREA,
        device=device
    )
    # handle None-safe
    if examplers is not None:
        examplers = examplers.to(device)

    with torch.no_grad():
        output, extra_out = model(
            image,
            cls_name_list,
            coop_require_grad=config['training'].get('coop_training', False),
            examplers=examplers
        )

    # batch loop (batch=1 in your current setup)
    batch_abs = 0.0
    batch_sq  = 0.0

    for i in range(output.shape[0]):
        pred_cnt = (output[i].sum() / SCALE_FACTOR).item()
        err = abs(pred_cnt - gt)
        batch_abs += err
        batch_sq  += err * err

        total_abs_err += err
        total_sq_err  += err * err
        total_samples += 1

    batch_mae  = batch_abs / output.shape[0]
    batch_rmse = math.sqrt(batch_sq / output.shape[0])

    print(f"[Eval] Image: {img_file}, GT count: {gt}, Pred count: {pred_cnt:.2f}, MAE: {batch_mae:.2f}, RMSE: {batch_rmse:.2f}")

epoch_mae  = total_abs_err / total_samples
epoch_rmse = math.sqrt(total_sq_err / total_samples)
logging.info(f"Eval epoch done | epoch MAE: {epoch_mae:.4f}, epoch RMSE: {epoch_rmse:.4f}")




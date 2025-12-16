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


warnings.filterwarnings("ignore")

SCALE_FACTOR = 60.0

def sliding_window(image, window_size = (384, 384), stride = 128):
    """
    Split an image into overlapping patches.
    Args:
        image: [3, 384, W], W >= 384
        window_size: (384, 384)
        stride: 128
    Returns:
        patches: [N, 3, 384, 384]
        intervals: [[start, end], [start, end], ...]
    """
    #right padding to make sure the image can be divided by stride
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    image = np.pad(image, ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)), 'constant')
    h, w, _ = image.shape
    assert h == 384, "FSC-147 assume image height is 384."
    patches = []
    intervals = []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])
    return np.array(patches).transpose(0,3,1,2), np.array(intervals)

def window_composite(patches, window_size = (384, 384), stride = 128):
    """
    Composite patches (from sliding window) into an image.
    for overlapping regions, average the values.
    Args:
        patches: [N, C, 384, 384]
        window_size: (384, 384)
        stride: the stride used in sliding window
    Returns:
        image: [1, 384, W ]
    """
    image = None
    patch_h, patch_w = window_size
    for i, patch in enumerate(patches):
        if i == 0:
            image = patch
            # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
            # cv2.imwrite(f"debug/out/image{i}.jpg", image)

        else:
            blend_width = patch_w - stride
            # cv2.imwrite(f"debug/out/patch{i}.jpg", patch)
            prev_to_blend = image[:, :, -blend_width:]
            # cv2.imwrite(f"debug/out/prev_to_blend{i}.jpg", prev_to_blend)
            next_to_blend = patch[:, :, :blend_width]
            # cv2.imwrite(f"debug/out/next_to_blend{i}.jpg", next_to_blend)
            blend_factor = torch.sigmoid(torch.tensor(np.linspace(-3, 3, blend_width))).to(image.device)
            blend = (1-blend_factor) * prev_to_blend + blend_factor * next_to_blend
            # cv2.imwrite(f"debug/out/blend{i}.jpg", blend)
            image[:, :, -blend_width:] = blend
            # cv2.imwrite(f"debug/out/image{i}.jpg", image)
            patch_remain = patch[:, :, blend_width:]
            #log all intermediate results
            image = torch.cat([image, patch_remain], dim = -1)
    return image


def extract_patches(img, patch_size=512, stride=512):
    _, _, h, w = img.size()
    num_h = (h - patch_size + stride - 1) // stride + 1
    num_w = (w - patch_size + stride - 1) // stride + 1
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            patch = img[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches, num_h, num_w

def reassemble_patches(patches, num_h, num_w, h, w, patch_size=512, stride=256):
    """
    patches: [N, C, ph, pw] hoặc [N, ph, pw]
    - Nếu ph,pw != patch_size: upscale để về patch_size (giữ count bằng cách chia area factor)
    - Nếu đã == patch_size: KHÔNG upscale
    """
    if patches.dim() == 3:          # [N,H,W] -> [N,1,H,W]
        patches = patches.unsqueeze(1)
    if patches.dim() != 4:
        raise ValueError(f"Expected 4D [N,C,H,W], got {patches.shape}")

    N, C, ph, pw = patches.shape

    # Nếu patch output chưa phải patch_size thì upscale về patch_size
    if (ph != patch_size) or (pw != patch_size):
        scale_y = patch_size / ph
        scale_x = patch_size / pw
        # FSC thường scale đều; nếu không đều thì vẫn xử lý được
        patches = F.interpolate(patches, size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # bảo toàn tổng count: diện tích tăng lên bao nhiêu thì chia lại bấy nhiêu
        area_factor = (patch_size * patch_size) / (ph * pw)
        patches = patches / area_factor

    result = torch.zeros(1, C, h, w, device=patches.device)
    norm_map = torch.zeros(1, 1, h, w, device=patches.device)

    patch_idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)

            # patches[patch_idx] là [C,patch,patch] -> add batch dim để broadcast sạch
            result[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += patches[patch_idx:patch_idx+1]
            norm_map[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += 1
            patch_idx += 1

    result = result / norm_map.clamp_min(1.0)
    return result


def get_exampler_tensor_stack(examplers, device):
    fixed = []
    for ex in examplers:    
        # ex: numpy HWC uint8 -> Tensor CHW float [0,1]
        ex_t = torch.from_numpy(ex).permute(2, 0, 1).contiguous().float() / 255.0  # (3,h,w)

        ex_t = VF.resize(ex_t, (384, 384))
        fixed.append(ex_t)

    examplers = torch.stack(fixed, dim=0).to(device)
    return examplers

class Engine:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = config
        self.model, self.optimizer, self.schedular = self.creates()
        self.loss = F.mse_loss
        self.rank_loss = RankLoss(temperature=0.07)
        self.current_epoch = 0
        self.contrast_pre_epoch = config['training'].get('contrast_pre_epoch', 20)
        self.min_mae = float('inf')
        self.get_exampler = GetExampler(device=self.device)

    def reload(self):
        checkpoint_path = self.config['training'].get('save_path', 'checkpoint.pth')
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.schedular.load_state_dict(checkpoint['schedular_state_dict'])
            logging.info("Reloaded model from {}".format(checkpoint_path))
        else:
            logging.info("No checkpoint path provided, training from scratch.")
        
        self.current_epoch = checkpoint.get('epoch', 0) if checkpoint_path is not None else 0
        self.min_mae = checkpoint.get('min_mae', float('inf')) if checkpoint_path is not None else float('inf')
        self.batch_size = self.config['training'].get('batch_size', 4)
    
    def save(self, epoch, score):
        checkpoint_path = self.config['training'].get('save_path', 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedular_state_dict': self.schedular.state_dict(),
            'min_mae': score
        }, checkpoint_path)

        logging.info("Saved model checkpoint to {}".format(checkpoint_path))
        
    def creates(self):
        model = LGCount(
            fim_depth=self.config['model'].get('fim_depth', 4),
            fim_num_heads=self.config['model'].get('fim_num_heads', 8),
            mlp_ratio=self.config['model'].get('mlp_ratio', 4.0)
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['optim']['lr'],
            betas=(0.9, 0.95),
            weight_decay=self.config['optim'].get('weight_decay', 0.05)
        )

        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.33)

        return model, optimizer, schedular
    

    def train(self, trainloader):
        self.model.train()

        progress_bar = tqdm(trainloader, desc="Training", leave=False)

        total_abs_err = 0.0      # tổng |pred - gt|
        total_sq_err  = 0.0      # tổng (pred - gt)^2
        total_samples = 0        # tổng số ảnh

        for batch in progress_bar:
            imgs = batch['image'].to(self.device)
            gt_density = batch['density'].to(self.device) * SCALE_FACTOR
            img_gd = batch['img_gd']
            img_src = batch['img_src']
            text = batch['text']

            examplers = self.get_exampler.get_highest_score_crop(img_gd, img_src, text, box_threshold=BOX_THRESHOLD, keep_area=KEEP_AREA, device=self.device).to(self.device)


            self.optimizer.zero_grad()

            output, extra_out = self.model(
                imgs,
                text,
                coop_require_grad=self.config['training'].get('coop_training', False),
                examplers=examplers
            )

            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (output.shape[0], 1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(self.device)

            # print(output.shape, gt_density.shape, masks.shape)
            # raise
            mse_loss = self.loss(output, gt_density)

            mse_loss = (mse_loss * masks / (384 * 384)).sum() / output.shape[0]

            rank_loss = self.rank_loss(
                extra_out['patch_embedding_contrast'],
                extra_out['class_text_embedding'],
                gt_density.detach().clone()
            )

            if self.contrast_pre_epoch >= self.current_epoch:

                loss = mse_loss + 0.01 * rank_loss
            else:
                loss = rank_loss
            # if self.current_epoch <= self.contrast_pre_epoch:
            #     loss = rank_loss 
                

            
            # Update information of MAE and RMSE
            batch_mae = 0

            batch_rmse = 0


            gt_sum = 0
            for i in range(output.shape[0]):
                pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()
                gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                gt_sum += gt_cnt
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2

                total_abs_err += cnt_err
                total_sq_err  += cnt_err ** 2
                total_samples += 1

            
            batch_mae /= output.shape[0]
            batch_rmse /= output.shape[0]
            batch_rmse = math.sqrt(batch_rmse)

            loss.backward()
            self.optimizer.step()
            # self.schedular.step()


            progress_bar.set_postfix(
                {
                    'batch_loss': loss.item(),
                    'batch_mae': batch_mae,
                    'batch_rmse': batch_rmse,
                    'lr' : self.optimizer.param_groups[0]['lr']
                })
        
        epoch_mae  = total_abs_err / total_samples
        epoch_rmse = math.sqrt(total_sq_err / total_samples)

        logging.info(
            "Train epoch done | epoch MAE: {:.4f}, epoch RMSE: {:.4f}".format(
                epoch_mae, epoch_rmse
            )
        )
    



            






    def evaluate(self, dataloader):
        batch_size = self.config['training'].get('batch_size', 4)
        self.model.eval()
        progress_bar = tqdm(dataloader, desc = "Evaluating", leave=False)

        total_abs_err = 0.0      # tổng |pred - gt|
        total_sq_err  = 0.0      # tổng (pred - gt)^2
        total_samples = 0        # tổng số ảnh
        epoch_res = []
        for batch in progress_bar:
            img = batch['image'].to(self.device)
            batch_count = batch['batch_cnt'].to(self.device)
            img_gd = batch['img_gd']
            img_src = batch['img_src']
            prompt = batch['text']

            imgs = img.to(self.device)
            text = prompt   

            cropped_imgs, num_h, num_w = extract_patches(imgs, patch_size=384,
                                                         stride=384)
            outputs = []
            Np = cropped_imgs.size(0)
            with torch.set_grad_enabled(False):
                examplers = self.get_exampler.get_highest_score_crop(img_gd, img_src, text, box_threshold=BOX_THRESHOLD, keep_area=KEEP_AREA, device=self.device).to(self.device)
            

                if examplers.size(0) == 1 and Np > 1:
                    examplers = examplers.repeat(Np, 1, 1, 1)
                elif examplers.size(0) != Np:
                    # fallback an toàn: nếu lệch vì lý do nào đó, ép đúng Np
                    examplers = examplers[:1].repeat(Np, 1, 1, 1)

                num_chunks = (cropped_imgs.size(0) + batch_size - 1) // batch_size
                for i in range(num_chunks):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, cropped_imgs.size(0))
                    # print(type(cropped_imgs[start_idx:end_idx]))
                    # raise
                    outputs_partial, _ = self.model(cropped_imgs[start_idx:end_idx], text * (end_idx - start_idx), coop_require_grad=self.config['training'].get('coop_training', False), examplers=examplers[start_idx:end_idx])
                    # đảm bảo out là [b, 1, h, w]
                    if outputs_partial.dim() == 3:      # [b,h,w]
                        outputs_partial = outputs_partial.unsqueeze(1)
                    outputs.append(outputs_partial)
                    
                results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, imgs.size(2), imgs.size(3),
                                             patch_size=384, stride=384)
                res = batch_count[0].item() - torch.sum(results).item() / 60
                epoch_res.append(res)
        
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
            
        logging.info(
            "Eval epoch done | epoch MAE: {:.4f}, epoch RMSE: {:.4f}".format(
                mae, mse
            )
        )
        return mae, mse
    
    def train_eval(self, train_loader, eval_loader):
        num_epochs = self.config['training']['num_epochs']
        reload_bool = self.config['training'].get('reload', False)
        if reload_bool:
            self.reload()

        
        
        current_epoch = self.current_epoch
        for epoch in range(current_epoch, num_epochs):
            self.current_epoch = epoch
            logging.info("Epoch {}/{}".format(epoch+1, num_epochs))
            self.train(train_loader)
            epoch_mae, epoch_rmse = self.evaluate(eval_loader)

            if epoch_mae < self.min_mae:
                self.min_mae = epoch_mae
                self.save(epoch+1, self.min_mae)
                logging.info("New best model saved with MAE: {:.4f}".format(self.min_mae))
            
            self.schedular.step()
    



if __name__ == "__main__":
    args = get_parser()
    config = load_config(args.config)
    logg(config['training']['log_file'])
    train_dataset = ObjectCount(
        config = config,
        split = "train"
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_train_object_count,
        num_workers=config['training'].get('num_workers', 4)
    )

    val_dataset = ObjectCount(
        config = config,
        split = "val"
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_test_object_count,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_dataset = ObjectCount(
        config = config,
        split = "test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_test_object_count,
        num_workers=config['training'].get('num_workers', 4)
    )

    engine = Engine(config)
    engine.train_eval(trainloader, val_loader)
    logging.info("Evaluating on test set...")
    engine.evaluate(test_loader)


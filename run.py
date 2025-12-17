# train_lgcount.py
# supress torchvision warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import os
import random
from pathlib import Path
import math
from PIL import Image

import torch
import torch.nn.functional as F
from typing import List, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
import einops
import cv2

import util.misc as misc
from util.FSC147 import FSC147
from util.CARPK import CARPK
from util.ShanghaiTech import ShanghaiTech
from models.rank_loss import RankLoss
from models.align_loss import AlignLoss
from models import LG_count
from util.constant import SCALE_FACTOR

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


def get_args_parser():
    parser = argparse.ArgumentParser('LG-Count', add_help=False)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    parser.add_argument('--backbone', default="b16", choices=["b16", "b32", "l14"], type=str)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--align_path', default='', type=str)
    parser.add_argument('--decoder_head', default=8, type=int)

    parser.add_argument('--use_mixed_fim', default=True, type=misc.str2bool)
    parser.add_argument('--unfreeze_vit', default=False, type=misc.str2bool)
    parser.add_argument('--use_fim', default=False, type=misc.str2bool)

    parser.add_argument('--use_coop', default=True, type=misc.str2bool)
    parser.add_argument('--coop_width', default=2, type=int)
    parser.add_argument('--coop_require_grad', default=False, type=misc.str2bool)

    parser.add_argument('--use_vpt', default=True, type=misc.str2bool)
    parser.add_argument('--vpt_width', default=20, type=int)
    parser.add_argument('--vpt_depth', default=10, type=int)

    parser.add_argument("--use_contrast", default=True, type=misc.str2bool)
    parser.add_argument("--w_contrast", default=1.0, type=float)
    parser.add_argument("--noise_text_ratio", default=0.0, type=float)
    parser.add_argument('--normalize_contrast', default=False, type=misc.str2bool)
    parser.add_argument('--contrast_pos', default="pre", choices=["pre", "post"], type=str)
    parser.add_argument('--contrast_pre_epoch', default=20, type=int)
    parser.add_argument('--start_val_epoch', default=100, type=int)

    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=0.)

    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--dataset_type', default="FSC", type=str, choices=["FSC", "CARPK", "COCO", "ShanghaiTech"])

    parser.add_argument('--output_dir', default='./out')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--log_dir', default='./out')
    parser.add_argument('--log_test_img', default=False, type=bool)
    parser.add_argument('--dont_log', action='store_true')
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--exp_note', default="", type=str)

    # =========================
    # Stage1 (RichCount Stage1)
    # =========================
    parser.add_argument('--stage1_ckpt', default='', type=str,
                        help='path to Stage1 checkpoint .pt; empty = disable')
    parser.add_argument('--use_stage1_ffn', default=True, type=misc.str2bool)
    parser.add_argument('--use_stage1_adapter', default=True, type=misc.str2bool)

    return parser


class Model(LightningModule):
    def __init__(self, args, all_classes: List[str] = None):
        super().__init__()
        self.args = args
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.all_classes = all_classes
        self.save_hyperparameters(args)

        stage1_ckpt_path = getattr(self.args, "stage1_ckpt", "")
        use_stage1_ffn = getattr(self.args, "use_stage1_ffn", True)
        use_stage1_adapter = getattr(self.args, "use_stage1_adapter", True)

        self.model = LG_count.LGCount(
            fim_depth=self.args.decoder_depth,
            fim_num_heads=self.args.decoder_head,
            use_coop=self.args.use_coop,
            use_vpt=self.args.use_vpt,
            coop_width=self.args.coop_width,
            vpt_width=self.args.vpt_width,
            vpt_depth=self.args.vpt_depth,
            backbone=self.args.backbone,
            use_fim=self.args.use_fim,
            use_mixed_fim=self.args.use_mixed_fim,
            unfreeze_vit=self.args.unfreeze_vit,
            contrast_pre_epoch=self.args.contrast_pre_epoch,

            # Stage1
            stage1_ckpt_path=stage1_ckpt_path,
            use_stage1_ffn=use_stage1_ffn,
            use_stage1_adapter=use_stage1_adapter,
            stage1_force_eval=True,
        )

        self.model_align = LG_count.LGCountAlign(
            fim_depth=self.args.decoder_depth,
            fim_num_heads=self.args.decoder_head,
            use_coop=self.args.use_coop,
            use_vpt=self.args.use_vpt,
            coop_width=self.args.coop_width,
            vpt_width=self.args.vpt_width,
            vpt_depth=self.args.vpt_depth,
            backbone=self.args.backbone,
            use_fim=self.args.use_fim,
            use_mixed_fim=self.args.use_mixed_fim,
            unfreeze_vit=self.args.unfreeze_vit,
            contrast_pre_epoch=self.args.contrast_pre_epoch,

            # Stage1
            stage1_ckpt_path=stage1_ckpt_path,
            use_stage1_ffn=use_stage1_ffn,
            use_stage1_adapter=use_stage1_adapter,
            stage1_force_eval=True,
        )

        self.loss = F.mse_loss
        self.rank_loss = RankLoss(0.07)
        self.align_loss = AlignLoss(0.07)
        self.loss_weight = 0.9

        # ---- load align prompt weights (your existing logic) ----
        ck = torch.load('./ckpt/epoch=149-avg_fine_accuracy_pred=0.71.ckpt', map_location="cpu")
        selected_weights = {}
        params = [
            'img_encoder.visual_prompt',
            'img_encoder.vpt_norm.weight',
            'img_encoder.vpt_norm.bias',
            'img_encoder.vpt_proj.weight',
            'img_encoder.vpt_proj.bias',
            'text_encoder.learnable_context'
        ]
        for k in params:
            selected_weights[k] = ck['state_dict']['model.' + k]

        self.model_align.load_state_dict(selected_weights, strict=False)
        self.model_align.requires_grad_(False)
        self.model_align.eval()

    def training_step(self, batch, batch_idx):
        samples, gt_density, boxes, m_flag, class_text, prompt_add, coarse_text_list, fine_text_list, coarse_GT, fine_GT, im_id = batch

        self.model_align.eval()
        pred_coarse_top_indices, pred_fine_top_indices, top_fine_text_embedding = self.model_align(
            samples, coarse_text_list, fine_text_list
        )

        output, extra_out = self.model(
            samples, class_text, self.current_epoch, top_fine_text_embedding,
            return_extra=True, coop_require_grad=True
        )

        mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
        masks = np.tile(mask, (output.shape[0], 1)).reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)

        mse_loss = self.loss(output, gt_density)
        mse_loss = (mse_loss * masks / (384 * 384)).sum() / output.shape[0]

        class_text_embedding = extra_out['class_text_embedding']
        patch_embedding_contrast = extra_out['patch_embedding_contrast']

        rank_loss = self.rank_loss(patch_embedding_contrast, class_text_embedding, gt_density.detach().clone())
        self.log('contrast_loss', rank_loss)
        self.log('mse_loss', mse_loss)

        loss = mse_loss + 0.01 * rank_loss
        if self.args.use_contrast and self.current_epoch <= self.args.contrast_pre_epoch:
            loss = rank_loss

        self.log('train_loss', loss)

        batch_mae = 0
        batch_rmse = 0
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()
            gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse / output.shape[0])

        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model_align.eval()

        image, gt_density, _, _, class_text, coarse_text_list, fine_text_list, im_id = batch
        if self.current_epoch < self.args.start_val_epoch:
            return {"mae": [20 + self.current_epoch], "rmse": [100], "prompt": coarse_text_list[0][0]}

        assert image.shape[0] == 1, "only support inference one image at a time"
        raw_h, raw_w = image.shape[2:]

        patches, _ = misc.sliding_window(image, stride=128)
        patches = torch.from_numpy(patches).float().to(self.device)

        class_text = np.repeat(class_text, patches.shape[0], axis=0)
        coarse_text_list = [[i] * patches.shape[0] for i in coarse_text_list]
        fine_text_list = [[[item] * patches.shape[0] for item in row] for row in fine_text_list]

        _, _, top_fine_text_embedding = self.model_align(patches, coarse_text_list, fine_text_list)
        output, _ = self.model(patches, class_text, self.current_epoch, top_fine_text_embedding)

        output = output.unsqueeze(1)
        output = misc.window_composite(output, stride=128)
        output = output.squeeze(1)[:, :, :raw_w]

        pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()
        gt_cnt = torch.sum(gt_density[0] / SCALE_FACTOR).item()
        cnt_err = abs(pred_cnt - gt_cnt)

        return {"mae": [cnt_err], "rmse": [cnt_err ** 2], "prompt": coarse_text_list[0][0][0]}

    def validation_epoch_end(self, outputs):
        all_mae, all_rmse = [], []
        for o in outputs:
            all_mae += o["mae"]
            all_rmse += o["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)
        self.logger.experiment.add_text("prompt", outputs[0]["prompt"], self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.args.unfreeze_vit:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("model.clip") or k.startswith("model.img_encoder.clip") or k.startswith("model.text_encoder.clip") or k.startswith("model.img_encoder.vit"):
                    del checkpoint["state_dict"][k]

        # remove Stage1 modules to keep ckpt small
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("model.stage1") or k.startswith("model_align.stage1"):
                del checkpoint["state_dict"][k]

    def overwrite_args(self, args):
        self.args = args


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_train = FSC147(split="train")
    all_classes_train = dataset_train.all_classes
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    dataset_val = FSC147(split="val", resize_val=False)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    save_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_mae', save_top_k=4, mode='min', filename='{epoch}-{val_mae:.2f}'
    )

    model = Model(args, all_classes=all_classes_train)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name)

    trainer = Trainer(
        accelerator="gpu",
        callbacks=[save_callback],
        accumulate_grad_batches=args.accum_iter,
        precision=16,
        max_epochs=args.epochs + args.contrast_pre_epoch,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
    )

    if args.mode == "train":
        if args.ckpt is not None:
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
        trainer.fit(model, train_dataloader, val_dataloader)

    elif args.mode == "test":
        if args.dataset_type == "FSC":
            dataset_val = FSC147(split="val", resize_val=False)
            dataset_test = FSC147(split="test")
        elif args.dataset_type == "COCO":
            dataset_val = FSC147(split="val_coco", resize_val=False)
            dataset_test = FSC147(split="test_coco")
        elif args.dataset_type == "CARPK":
            dataset_val = dataset_test = CARPK(None, split="test")
        elif args.dataset_type == "ShanghaiTech":
            dataset_val = dataset_test = ShanghaiTech(None, split="test", part="B")
        else:
            raise ValueError("Unknown dataset_type")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=1,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test, batch_size=1,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )

        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test")

        model = Model.load_from_checkpoint(args.ckpt, strict=False)
        model.overwrite_args(args)
        model.eval()

        if args.dataset_type in ["FSC", "COCO"]:
            print("====Metric on val set====")
            trainer.test(model, val_dataloader)
        print("====Metric on test set====")
        trainer.test(model, test_dataloader)

from datasets import *
import torch
from tqdm import tqdm
from models import *
import numpy as np
import torch.nn.functional as F
import warnings
import logging

warnings.filterwarnings("ignore")

SCALE_FACTOR = 60.0

class Engine:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = config
        self.model, self.optimizer, self.schedular = self.creates()
        self.loss = F.mse_loss
        self.rank_loss = RankLoss(temperature=0.07)
    
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
    
    def save(self, epoch):
        checkpoint_path = self.config['training'].get('save_path', 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'schedular_state_dict': self.schedular.state_dict(),
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

        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)

        return model, optimizer, schedular
    

    def train(self, trainloader):
        self.model.train()
        progress_bar = tqdm(trainloader, desc="Training", leave=False)
        for batch in progress_bar:
            imgs = batch['image'].to(self.device)
            gt_density = batch['density'].to(self.device)
            boxes = batch['boxes']
            m_flag = batch['m_flag']
            text = batch['text']

            self.optimizer.zero_grad()

            output, extra_out = self.model(
                imgs,
                text,
                coop_require_grad=self.config['training'].get('coop_training', False)
            )

            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (output.shape[0], 1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(self.device)
            mse_loss = self.loss(output, gt_density)

            mse_loss = (mse_loss * masks / (384 * 384)).sum() / output.shape[0]

            rank_loss = self.rank_loss(
                extra_out['patch_embedding_contrast'],
                extra_out['class_text_embedding'],
                gt_density.detach().clone()
            )

            loss = mse_loss + 0.01 * rank_loss
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
            batch_mae /= output.shape[0]
            batch_rmse /= output.shape[0]
            batch_rmse = math.sqrt(batch_rmse)

            loss.backward()
            self.optimizer.step()
            self.schedular.step()

            logging.info("Loss: {:.4f}, MSE Loss: {:.4f}, Rank Loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, GT Sum: {:.4f}".format(
                loss.item(), mse_loss.item(), rank_loss.item(), batch_mae, batch_rmse, gt_sum
            ))

            progress_bar.set_postfix(
                batch_loss = loss.item())

            






    def evaluate(self, dataloader):
        self.model.eval()
        progress_bar = tqdm(dataloader, desc = "Evaluating", leave=False)
        for batch in progress_bar:
            imgs = batch['image'].to(self.device)
            gt_density = batch['density'].to(self.device)
            boxes = batch['boxes']
            m_flag = batch['m_flag']
            text = batch['text']

            with torch.no_grad():
                output, extra_out = self.model(
                    imgs,
                    text,
                    coop_require_grad=False
                )


            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (output.shape[0], 1))
            masks = masks.reshape(output.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(self.device)
            mse_loss = self.loss(output, gt_density)

            mse_loss = (mse_loss * masks / (384 * 384)).sum() / output.shape[0]

            rank_loss = self.rank_loss(
                extra_out['patch_embedding_contrast'],
                extra_out['class_text_embedding'],
                gt_density.detach().clone()
            )

            loss = mse_loss + 0.01 * rank_loss
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
            batch_mae /= output.shape[0]
            batch_rmse /= output.shape[0]
            batch_rmse = math.sqrt(batch_rmse)

            logging.info("Loss: {:.4f}, MSE Loss: {:.4f}, Rank Loss: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, GT Sum: {:.4f}".format(
                loss.item(), mse_loss.item(), rank_loss.item(), batch_mae, batch_rmse, gt_sum
            ))

            progress_bar.set_postfix(
                batch_loss = loss.item())

            return batch_mae, batch_rmse
    
    def train_eval(self, train_loader, eval_loader):
        num_epochs = self.config['training']['num_epochs']
        reload_bool = self.config['training'].get('reload', False)
        if reload_bool:
            self.reload()

        min_mae = float('inf')

        for epoch in range(num_epochs):
            logging.info("Epoch {}/{}".format(epoch+1, num_epochs))
            self.train(train_loader)
            batch_mae, batch_rmse = self.evaluate(eval_loader)

            if batch_mae < min_mae:
                min_mae = batch_mae
                self.save(epoch+1)
                logging.info("New best model saved with MAE: {:.4f}".format(min_mae))
            
    



if __name__ == "__main__":
    args = get_parser()
    config = load_config(args.config)

    dataset = FSC147(
        config = config,
        split = "train"
    )
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    val_dataset = FSC147(
        config = config,
        split = "val"
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_dataset = FSC147(
        config = config,
        split = "test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    engine = Engine(config)
    engine.train_eval(trainloader, val_loader)
    logging.info("Evaluating on test set...")
    engine.evaluate(test_loader)


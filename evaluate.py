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
from run import Engine

if __name__ == "__main__":
    args = get_parser()
    config = load_config(args.config)
    logg(config['training']['log_file'])

    
    test_dataset = FSC147(
        config = config,
        split = "test",
        subset_scale=1.0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )

    engine = Engine(config)
    logging.info("Evaluating on test set...")
    engine.eval_batch(test_loader)

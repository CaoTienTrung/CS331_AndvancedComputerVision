from dataset import *
import torch
from tqdm import tqdm


import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = get_parser()
    config = load_config(args.config)

    dataset = FSC147(
        config = config,
        split = "train"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    for batch in tqdm(dataloader, desc = "Testing DataLoader"):
        image = batch['image']
        density = batch['density']
        boxes = batch['boxes']
        m_flag = batch['m_flag']
        text = batch['text']

        print("Image shape:", image.shape)
        print("Density shape:", density.shape)
        print("Boxes:", boxes)
        print("Mask flag:", m_flag.shape)
        print("Text prompts:", text)
        break  # Just test one batch


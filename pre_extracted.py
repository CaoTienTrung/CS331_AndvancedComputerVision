from datasets import *
from tqdm import tqdm 

def test_predict_batch():
    import time 
    # image_path = "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg"
    img_list = [
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg",
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/272.jpg",
    ]
    get_exampler = GetExampler()
    
    curr = time.time()
    boxes, logits,  img_sources = get_exampler.predict_batch(
        imag_paths=img_list,
        captions=["strawberry", "penguins"]
    )

    for i in range(len(img_list)):

        annotated = get_exampler.annotate(image_source=img_sources[i], boxes=boxes[i], logits=logits[i])
        out_path = f"/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino_{i}.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)
    print("Time per batch:", (time.time() - curr))


class PreExtractedGetExampler(Dataset):
    def __init__(self, config,
                 split:str , subset_scale: float = 1.0):
        """
        Parameters
        ----------
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']

        #!HARDCODED Dec 25: 
        self.data_dir = config['training']['data_dir'] 
        self.dataset_type = config['training']['dataset_type'] # FSC147 
        additional_prompt = config['training'].get('additional_prompt', False)
        subset_scale = subset_scale

        self.resize_val = config['training']['resize_val'] if split == 'val' else False

        self.im_dir = os.path.join(self.data_dir,'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir,  f'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(self.data_dir, f'Train_Test_Val_FSC_147.json')
        self.class_file = os.path.join(self.data_dir,f'ImageClasses_FSC147.txt')
        self.split = split
        with open(self.data_split_file) as f:
            data_split = json.load(f)

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        self.idx_running_set = data_split[split]
        # subsample the dataset
        self.idx_running_set = self.idx_running_set[:int(subset_scale*len(self.idx_running_set))]

        self.class_dict = {}
        with open(self.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                # concat word as string
                val = ' '.join(val)
                self.class_dict[key] = val
        self.all_classes = list(set(self.class_dict.values()))
    
    def __len__(self):
        return len(self.idx_running_set)
    
    def __getitem__(self, idx):
        img_id = self.idx_running_set[idx]
        img_path = os.path.join(self.im_dir, f'{img_id}')

        class_name = self.class_dict[img_id]

        sample = {
            'class_name': class_name,
            'img_id': img_id,
            'img_path': img_path,
        }
        return sample
    
def collate_fn(batch):
    class_names = [item['class_name'] for item in batch]
    img_ids = [item['img_id'] for item in batch]
    img_paths = [item['img_path'] for item in batch]


    return {
        'class_names': class_names,
        'img_ids': img_ids,
        'img_paths': img_paths,
    }

def extract_to_path():
    dataset = PreExtractedGetExampler(
        config = {
            'training': {
                'data_dir': '/home/anhkhoa/anhkhoa/CountingObject/Dataset',
                'dataset_type': 'FSC147',
            }
        },
        split = "test",
        subset_scale=1.0
    )
    save_path = '/home/anhkhoa/anhkhoa/CountingObject/pre_extracted_path'
    os.makedirs(save_path, exist_ok=True)

    get_exampler = GetExampler()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn = collate_fn,
        num_workers=4,
    )

    progress_bar = tqdm(loader, desc="Extracting crops")
    for batch in progress_bar:
        class_names = batch['class_names']
        img_ids = batch['img_ids']
        img_paths = batch['img_paths']
    
        crops = get_exampler.get_highest_score_crop_img_path_ver(
            img_path=img_paths,
            captions=class_names,
            box_threshold=0,
            keep_area=0.1
        )

        for i in range(len(img_ids)):
            crop = crops[i]
            img_id = img_ids[i]
            out_path = os.path.join(save_path, f'{img_id}_crop.jpg')
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            print("Saved:", out_path)
            progress_bar.set_postfix({'last_saved': out_path})
    

if __name__ == "__main__":
    # test()
    # test_normal()
    # test_crop_best()
    # test_predict_batch()
    # test_crop_best()
    extract_to_path()
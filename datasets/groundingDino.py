import sys
import os
HOME= os.getcwd()
path = os.path.abspath(os.path.join(HOME, 'CountingObject/datasets/GroundingDINO'))
sys.path.append(path)
import torch 
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

from .GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)
from .GroundingDINO.groundingdino.datasets import transforms as T
from PIL import Image

import supervision as sv
import numpy as np
from torchvision.ops import box_convert
from typing import List, Tuple
from torchvision import transforms

DIR_WEIGHTS = os.path.join(HOME, "CountingObject/datasets/pretrained_models")
CONFIG_PATH = os.path.join(HOME, "CountingObject/datasets/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(DIR_WEIGHTS, "groundingdino_swint_ogc.pth")



class GetExampler:
    def __init__(self, device='cuda'):
        self.download_model()
        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH, device)
        self.model = self.model.to(device)


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def download_model(self):
        os.makedirs(DIR_WEIGHTS, exist_ok=True)
        if os.path.exists(WEIGHTS_PATH):
            print(f"Model weights already exist at {WEIGHTS_PATH}. Skipping download.")
        else:
            import urllib.request
            print("Downloading model weights...")
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, WEIGHTS_PATH)
            print("Saved to:", WEIGHTS_PATH)



    def get_exampler(self, image_path, caption, box_threshold=0.35, text_threshold=0.25, device='cuda'):
        imag_source, image_transformed = load_image(image_path)
        boxes, logits, phrases = predict(
            self.model,
            image_transformed,
            caption,
            box_threshold,
            text_threshold,
            device,
            remove_combined=False
        )
        return boxes, logits, phrases, imag_source
    

    def annotate(self, image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
        """    
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (np.ndarray): The source image to be annotated.
        boxes (torch.Tensor): A tensor containing bounding box coordinates.
        logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
        phrases (List[str]): A list of labels for each bounding box.

        Returns:
        np.ndarray: The annotated image.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)


        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
        return annotated_frame
    
    def predict_batch_img_path_ver(self, imag_paths, captions, box_threshold=0.35, device='cuda'):
        images = []
        img_sources = []
        for img_path in imag_paths:
            imag_source, image_transformed = load_image(img_path)
            images.append(image_transformed)      # Tensor (3,H,W) float
            img_sources.append(imag_source)       # numpy HWC

        captions = [preprocess_caption(c) for c in captions]

        # IMPORTANT: pad thành NestedTensor thay vì stack
        from .GroundingDINO.groundingdino.util.misc import nested_tensor_from_tensor_list
        samples = nested_tensor_from_tensor_list(images).to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(samples, captions=captions)

        pred_logits = outputs["pred_logits"].sigmoid().detach().cpu()
        pred_boxes  = outputs["pred_boxes"].detach().cpu()

        batch_boxes, batch_scores = [], []
        B = pred_logits.shape[0]
        for b in range(B):
            scores = pred_logits[b].max(dim=1)[0]
            keep = scores > box_threshold
            batch_boxes.append(pred_boxes[b][keep])
            batch_scores.append(scores[keep])

        return batch_boxes, batch_scores, img_sources

    
    def get_highest_score_crop_img_path_ver(self, img_path, captions, box_threshold=0.35, keep_area=0.4, device='cuda'):
        boxes_list, scores_list, imag_source = self.predict_batch_img_path_ver(
            imag_paths=img_path,
            captions=captions,
            box_threshold=box_threshold,
            device=device
        )

        batch_crops = []
        for i in range(len(img_path)):
            boxes = boxes_list[i]   # (N,4) normalized cxcywh
            scores = scores_list[i] # (N,)

            if boxes.numel() == 0:
                batch_crops.append(None)
                continue

            H, W, _ = imag_source[i].shape
            img_area = float(W * H)

            # 1) scale -> pixel
            scale = torch.tensor([W, H, W, H], dtype=boxes.dtype)
            boxes_pix = boxes * scale  # (N,4) pixel cxcywh

            # 2) cxcywh -> xyxy
            xyxy = box_convert(boxes=boxes_pix, in_fmt="cxcywh", out_fmt="xyxy")  # (N,4)

            # 3) clamp vào biên ảnh
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W - 1)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H - 1)

            # 4) tính area ratio
            bw = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
            bh = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
            area = bw * bh
            area_ratio = area / img_area

            # 5) lọc theo keep_area + bỏ box “rỗng”
            keep = (area_ratio <= keep_area) & (bw > 1) & (bh > 1)

            boxes_pix = boxes_pix[keep]
            xyxy = xyxy[keep]
            scores = scores[keep]

            if scores.numel() == 0:
                batch_crops.append(None)
                continue

            # 6) giờ mới lấy max trên tập đã lọc
            max_idx = torch.argmax(scores).item()
            x1, y1, x2, y2 = xyxy[max_idx].int().tolist()

            crop = imag_source[i][y1:y2, x1:x2, :]
            batch_crops.append(crop)

        return batch_crops
    

    def predict_batch(self, images, captions, box_threshold=0.35, device='cuda'):
        # images = []
        # img_sources = []
        # for img_path in imag_paths:
        #     imag_source, image_transformed = load_image(img_path)
        #     images.append(image_transformed)      # Tensor (3,H,W) float
        #     img_sources.append(imag_source)       # numpy HWC


        captions = [preprocess_caption(c) for c in captions]

        # IMPORTANT: pad thành NestedTensor thay vì stack
        from .GroundingDINO.groundingdino.util.misc import nested_tensor_from_tensor_list
        samples = nested_tensor_from_tensor_list(images).to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(samples, captions=captions)

        pred_logits = outputs["pred_logits"].sigmoid().detach().cpu()
        pred_boxes  = outputs["pred_boxes"].detach().cpu()

        batch_boxes, batch_scores = [], []
        B = pred_logits.shape[0]
        for b in range(B):
            scores = pred_logits[b].max(dim=1)[0]
            keep = scores > box_threshold
            batch_boxes.append(pred_boxes[b][keep])
            batch_scores.append(scores[keep])

        return batch_boxes, batch_scores
    
    def get_highest_score_crop(self, images, image_sources, captions, box_threshold=0.35, keep_area=0.4, device='cuda'):
        boxes_list, scores_list = self.predict_batch(
            images=images,
            captions=captions,
            box_threshold=box_threshold,
            device=device
        )

        batch_crops = []
        for i in range(len(images)):
            boxes = boxes_list[i]   # (N,4) normalized cxcywh
            scores = scores_list[i] # (N,)

            if boxes.numel() == 0:
                batch_crops.append(None)
                continue

            H, W, _ = image_sources[i].shape
            img_area = float(W * H)

            # 1) scale -> pixel
            scale = torch.tensor([W, H, W, H], dtype=boxes.dtype)
            boxes_pix = boxes * scale  # (N,4) pixel cxcywh

            # 2) cxcywh -> xyxy
            xyxy = box_convert(boxes=boxes_pix, in_fmt="cxcywh", out_fmt="xyxy")  # (N,4)

            # 3) clamp vào biên ảnh
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W - 1)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H - 1)

            # 4) tính area ratio
            bw = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
            bh = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
            area = bw * bh
            area_ratio = area / img_area

            # 5) lọc theo keep_area + bỏ box “rỗng”
            keep = (area_ratio <= keep_area) & (bw > 1) & (bh > 1)

            boxes_pix = boxes_pix[keep]
            xyxy = xyxy[keep]
            scores = scores[keep]

            if scores.numel() == 0:
                batch_crops.append(None)
                continue

            # 6) giờ mới lấy max trên tập đã lọc
            max_idx = torch.argmax(scores).item()
            x1, y1, x2, y2 = xyxy[max_idx].int().tolist()
            crop = image_sources[i][y1:y2, x1:x2, :]

            # resize crop
            crop = cv2.resize(crop, (384, 384))
            crop = self.transform(Image.fromarray(crop))

            batch_crops.append(crop)
        
        # print("Type of batch crops:", type(batch_crops[0]))
        return torch.stack(batch_crops, dim = 0) # to do : transform crop img truoc khi vao model 


        
    # 1 function de return : anh co score cao nhat da crop


    
    

        
import cv2

def test_normal():
    import time 
    image_path = "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg"
    get_exampler = GetExampler()
    
    curr = time.time()
    boxes, logits, phrases, img_source = get_exampler.get_exampler(
        image_path=image_path,
        caption="strawberry"
    )
    annotated = annotate(image_source=img_source, boxes=boxes, logits=logits, phrases=phrases)
    out_path = f"/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)
    print("Time:", (time.time() - curr))

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
        image_paths=img_list,
        captions=["strawberry", "penguins"]
    )

    for i in range(len(img_list)):

        annotated = get_exampler.annotate(image_source=img_sources[i], boxes=boxes[i], logits=logits[i])
        out_path = f"/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino_{i}.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)
    print("Time per batch:", (time.time() - curr))


def test_crop_best():
    import time 
    image_path = "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg"

    img_list = [
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg",
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/272.jpg",
    ]
    get_exampler = GetExampler()
    
    curr = time.time()
    crops = get_exampler.get_highest_score_crop(
        image_paths=img_list,
        captions=["strawberry", "penguins"]
    )

    for i in range(len(img_list)):
        crop = crops[i]
        if crop is not None:
            out_path = f"/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino_crop_best_{i}.jpg"
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            print("Saved:", out_path)
        else:
            print("No box detected for image", i)
    print("Time per batch:", (time.time() - curr))

# if __name__ == "__main__":
#     # test()
#     # test_normal()
#     # test_crop_best()
#     test_predict_batch()
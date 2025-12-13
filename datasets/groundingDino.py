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

from GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)
import supervision as sv
import numpy as np
from torchvision.ops import box_convert
from typing import List, Tuple

DIR_WEIGHTS = os.path.join(HOME, "CountingObject/datasets/pretrained_models")
CONFIG_PATH = os.path.join(HOME, "CountingObject/datasets/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(DIR_WEIGHTS, "groundingdino_swint_ogc.pth")

class GetExampler:
    def __init__(self, device='cuda'):
        self.download_model()
        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH, device)
        self.model = self.model.to(device)

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
    
    def predict_batch(self, image_paths, captions, box_threshold=0.35, text_threshold=0.25, device='cuda'):
        image = []
        image_source = []
        for image_path in image_paths:
            img_src, img_transformed = load_image(image_path)
            image_source.append(img_src)
            image.append(img_transformed)

        images = torch.stack(image, dim=0).to(device)
        captions = [preprocess_caption(caption) for caption in captions]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images, captions=captions)

        # outputs["pred_logits"]: (B, nq, 256)
        # outputs["pred_boxes"] : (B, nq, 4)
        pred_logits = outputs["pred_logits"].sigmoid().detach().cpu()
        pred_boxes  = outputs["pred_boxes"].detach().cpu()

        batch_boxes, batch_scores = [], []
        B = pred_logits.shape[0]
        for b in range(B):
            scores = pred_logits[b].max(dim=1)[0]          # (nq,)
            keep = scores > box_threshold
            batch_boxes.append(pred_boxes[b][keep])        # (n_keep,4)
            batch_scores.append(scores[keep])              # (n_keep,)

        return batch_boxes, batch_scores, image_source
    
    

        
import cv2

def test():
    import time 
    image_path = "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg"
    img_list = [
        image_path,
        image_path,
    ]
    get_exampler = GetExampler()
    
    curr = time.time()
    boxes, logits,  img_sources = get_exampler.predict_batch(
        image_paths=img_list,
        captions=["strawberry", "strawberry"]
    )

    for i in range(len(img_list)):

        annotated = get_exampler.annotate(image_source=img_sources[i], boxes=boxes[i], logits=logits[i])
        out_path = f"/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino_{i}.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print("Saved:", out_path)
    print("Time per batch:", (time.time() - curr))
if __name__ == "__main__":
    test()

    
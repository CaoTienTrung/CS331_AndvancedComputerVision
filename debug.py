from datasets import *


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


def test_crop_best():
    import time 

    img_list = [
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg",
        "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/272.jpg",
    ]
    get_exampler = GetExampler()
    
    curr = time.time()
    crops = get_exampler.get_highest_score_crop_img_path_ver(
        img_path=img_list,
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

if __name__ == "__main__":
    # test()
    # test_normal()
    # test_crop_best()
    # test_predict_batch()
    test_crop_best()
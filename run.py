import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import torchvision.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os
import sys
import argparse
import warnings

# Load GroundingDINO model
def load_groundingdino_model():
    config_path = "GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "groundingdino_swint_ogc.pth"  # Path to pre-trained model weights
    model = load_model(config_path, checkpoint_path)
    return model


def get_groundingdino_bounding_box(image_path, text_prompt):
    # Load the GroundingDINO model
    model = load_groundingdino_model()
    image_source, image = load_image(image_path)
    # print(image)

    # Make predictions using the model
    boxes,logits,phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=0.3,  # Adjust this threshold for more/less boxes
        text_threshold=0.25  # Adjust this threshold based on text relevance
    )
 
    # Show the image with bounding boxes
    # annotated_image = annotate(image, boxes, phrases)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # plot_image_with_boxes(image, boxes, phrases)
    cv2.imwrite(f"Output/annotated/{phrases}.jpg", annotated_frame)
    return boxes, image

def build_sam() :
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    return predictor

# Helper function to load an image
def upload_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # plt.imshow(image)
    # plt.show()
    return image

def show_mask(mask, image,path, color=[255, 0, 0], transparency=0.5):
    mask = mask.squeeze()  # Remove extra dimensions if any
    red_mask = np.zeros_like(image)
    red_mask[:, :] = color  # Set color for the mask

    masked_image = np.copy(image)
    masked_image[mask > 0] = ((1 - transparency) * masked_image[mask > 0] + transparency * red_mask[mask > 0]).astype(np.uint8)
    
    # print(type(masked_image))
    array=masked_image
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = np.clip(array, 0, 1)  # Ensure the values stay between [0, 1]
    elif array.dtype == np.uint8:
        array = array / 255.0  # Normalize to [0, 1] for floats

    matplotlib.image.imsave(path, array)

    
    
   
    return masked_image


# Step 2: Use SAM for precise segmentation based on GroundingDINO's bounding box
def segment_with_sam(predictor,image_path, boxes):
    # Load SAM model
    image = upload_image(image_path)
    image_np = np.array(image)
    image_np = image_np.astype(np.float32) / 255.0
    # print("done upload")
    # predictor = load_sam_model()
    # Set the image in SAM predictor
    predictor.set_image(image_np)
    # print("done 4")

  
    W, H = image.size
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = cx - (w / 2)
    y_min = cy - (h / 2)
    x_max = cx + (w / 2)
    y_max = cy + (h / 2)
    # Stack and multiply by image dimensions to get pixel scale
    adj_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1) * torch.tensor([W, H, W, H])
    input_box=adj_boxes[0].cpu().numpy()
    masks, _, _ = predictor.predict(box=input_box, point_coords=None, point_labels=None, multimask_output=False)

    return image,masks,image_np


def save_images(folder_name,image,mask) :
    os.makedirs(f"output/{folder_name}", exist_ok=True)
    
    image.save(f"output/{folder_name}/Original_Image.jpg")
    mask_np = (mask[0] > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_np)
    mask_image.save(f"output/{folder_name}/mask.png")



def parse_args():
    parser = argparse.ArgumentParser(description="Object Segmentation and Inpainting with Stable Diffusion")
    
    # Add arguments
    parser.add_argument('--image', type=str, required=True, help='Path to the input image (e.g., ./example.jpg)')
    parser.add_argument('--class', type=str, required=True, help='Class of the object to segment (e.g., shelf)')


    #task1
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image (e.g., ./generated.png)')

    #task2
    parser.add_argument('--x', type=int, help='X-axis shift (e.g., 80)')
    parser.add_argument('--y', type=int, help='Y-axis shift (e.g., 0)')

    return parser.parse_args()
# Test the function
def main():

    args = parse_args()
    warnings.filterwarnings("ignore")
    if args.output is not None:
        image_path = args.image
        text_prompt = args.__dict__['class']
        output_path = args.output
        boxes,_ = get_groundingdino_bounding_box(image_path, text_prompt)
        predictor=build_sam()
        image,mask,image_np = segment_with_sam(predictor,image_path, boxes)
        show_mask(mask,image_np,output_path)
        save_images(text_prompt,image,mask)
    
    elif args.x is not None and args.y is not None:
        pass
    else:
        print("Invalid command. Either provide '--x' and '--y' or '--output'.")
 

    # image_path = "examples/bagpack.jpg" # Specify the image path
    
    # text_prompt = "laptop"  # Specify the object to locate (e.g., "dog")
    
    # boxes,_ = get_groundingdino_bounding_box(image_path, text_prompt)
    # print(boxes[0])
    # predictor=build_sam()
    # mask,image,segmented_image,_ = segment_with_sam(predictor,image_path, boxes)
    # save_images("bagpack",image,mask)
    
if __name__ == "__main__":
    main()
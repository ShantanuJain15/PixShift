import os
import sys
import argparse
import warnings
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2
import torchvision.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline


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

def show_mask(mask, image,path=None, color=[255, 0, 0], transparency=0.5):
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

    if path is not None :
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


def save_images(folder_name,image,mask,parent="output") :

    os.makedirs(f"{parent}/{folder_name}", exist_ok=True)
    
    image.save(f"{parent}/{folder_name}/Original_Image.jpg")
    mask_np = (mask[0] > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_np)
    mask_image.save(f"{parent}/{folder_name}/mask.png")


def extract_object(original_img, mask_img):
    object_img = cv2.bitwise_and(original_img, original_img, mask=mask_img)
    return object_img

# Step 2: Shift the object by applying an affine transformation
def shift_object(object_img, mask_img, x_shift, y_shift):
    rows, cols, _ = object_img.shape
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])  # Affine transform matrix
    shifted_object_img = cv2.warpAffine(object_img, M, (cols, rows))
    shifted_mask_img = cv2.warpAffine(mask_img, M, (cols, rows))
    return shifted_object_img, shifted_mask_img






def parse_args():
    parser = argparse.ArgumentParser(description="Object Segmentation and Inpainting with Stable Diffusion")
    
    # Add arguments
    parser.add_argument('--image', type=str, required=True, help='Path to the input image (e.g., ./example.jpg)')
    parser.add_argument('--class', type=str, required=True, help='Class of the object to segment (e.g., shelf)')


    #task1
    parser.add_argument('--output', type=str, help='Path to save the output image (e.g., ./generated.png)')

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
        image_path = args.image
        text_prompt = args.__dict__['class']
        x=args.x
        y=args.y
        boxes,_ = get_groundingdino_bounding_box(image_path, text_prompt)
        predictor=build_sam()
        image,mask,image_np = segment_with_sam(predictor,image_path, boxes)
        save_images(text_prompt,image,mask,"temp")

        original_img = cv2.imread(f'temp/{text_prompt}/Original_Image.jpg')
        mask_img = cv2.imread(f'temp/{text_prompt}/mask.png', cv2.IMREAD_GRAYSCALE)  


        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
       
        if mask_img.shape != original_img.shape:
            mask_img = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]))
        
        object_img = extract_object(original_img, mask_img)
        shifted_object_img, shifted_mask_img = shift_object(object_img, mask_img, x, y)

        os.makedirs(f'Output/task2{text_prompt}', exist_ok=True)
        cv2.imwrite(f'Output/task2{text_prompt}/shifted_mask.png',shifted_mask_img )
        cv2.imwrite(f'Output/task2{text_prompt}/shifted_object.png',shifted_object_img )
        cv2.imwrite(f'Output/task2{text_prompt}/object.png',object_img)
    
# Step 3: Use inpainting to remove the object from its original location
# Convert images from OpenCV (BGR) to PIL (RGB) for Stable Diffusion
        # inpainted_image_cv2 = np.array(inpainted_image_pil)
        # inpainted_image_cv2 = cv2.cvtColor(inpainted_image_cv2, cv2.COLOR_RGB2BGR)
        # original_img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        # mask_img_pil = Image.fromarray(mask_img)
        # Create an inverse mask of the shifted mask
        
        inverse_mask = cv2.bitwise_not(shifted_mask_img)
        
        if len(inverse_mask.shape) == 3:
            inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_BGR2GRAY)
        

# Ensure the mask is binary (0 or 255)
        _, inverse_mask = cv2.threshold(inverse_mask, 127, 255, cv2.THRESH_BINARY)
        

# Resize the mask to match the inpainted image dimensions, if necessary
        # if inverse_mask.shape != inpainted_image_cv2.shape[:2]:
        #     inverse_mask = cv2.resize(inverse_mask, (inpainted_image_cv2.shape[1], inpainted_image_cv2.shape[0]))

# Now use the inverse mask to remove the area from the background where the object will be placed
        
        background_with_hole = cv2.bitwise_and(original_img, original_img, mask=inverse_mask)
        cv2.imwrite(f'Output/task2{text_prompt}/black_hole.png', background_with_hole)
        
        # Ensure the shifted object and background have the same size
        if shifted_object_img.shape[:2] != background_with_hole.shape[:2]:
            shifted_object_img = cv2.resize(shifted_object_img, (background_with_hole.shape[1], background_with_hole.shape[0]))

# Ensure both images have the same number of channels   
        if len(shifted_object_img.shape) == 2:  # if the object is grayscale (single channel)
            shifted_object_img = cv2.cvtColor(shifted_object_img, cv2.COLOR_GRAY2BGR)  # convert to 3-channel BGR

        if len(background_with_hole.shape) == 2:  # if the background is grayscale (single channel)
            background_with_hole = cv2.cvtColor(background_with_hole, cv2.COLOR_GRAY2BGR)  # convert to 3-channel BGR

# Now add the shifted object to the new background
        final_image = cv2.add(background_with_hole, shifted_object_img)

# Save or display the result
        cv2.imwrite(f'Output/task2{text_prompt}/shifted.png',final_image )
        cv2.imwrite('shifted_object_composite.png', final_image)


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
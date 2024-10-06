# PixShift




### Prerequisites

- Python 3.11
- WGET
- Cuda

## Installation

Install python packages via commands:
```
pip3 install -r requirements.txt
```
Download pretrained model weights
```
cd PROJECT_ROOT_DIR
bash scripts/download_model.sh
```
or run this python code on project_dir for downloading the pre-trained model
```
import requests

# URL of the SAM model file you want to download
sam_model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Specify the output file path where you want to save the downloaded file
sam_output_file = "sam_vit_h_4b8939.pth"

# Send a GET request to the URL to download the SAM model file
response = requests.get(sam_model_url)

# Save the content of the response to the specified file
with open(sam_output_file, "wb") as f:
    f.write(response.content)

print(f"SAM model downloaded successfully and saved as {sam_output_file}")

# URL of the file you want to download
url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# Specify the output file path where you want to save the downloaded file
output_file = "groundingdino_swint_ogc.pth"

# Send a GET request to the URL to download the file
response = requests.get(url)

# Save the content of the response to the specified file
with open(output_file, "wb") as f:
    f.write(response.content)

print(f"File downloaded successfully and saved as {output_file}")

# URL of the file you want to download
url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# Specify the output file path where you want to save the downloaded file
output_file = "GroundingDINO_SwinT_OGC.py"

# Send a GET request to the URL to download the file
response = requests.get(url)

# Save the content of the response to the specified file
with open(output_file, "wb") as f:
    f.write(response.content)

print(f"File downloaded successfully and saved as {output_file}")
```

 ## Command for TASK 1
```
python run.py --image ./example.jpg --class shelf --output ./generated.png
```

 ## Command for TASK 2
```
python run.py --image ./example.jpg --class shelf --x 80 —-y 0
```




# TASK 1

## Segmentating the Object using text prompt

Using GroundingDino, we are making a Box around the object,which is mentioned in the task, then using Segement-Anything(SAM) we segment the object inside the box. SAM gives us mask image of the object and then we create red mask on the object using mask and original image.

 PROMPT = "BAGPACK"  

 Detect the object based on text prompt using [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 
 ![Box on the object](/Output/annotated/['bagpack'].jpg)

 Masking of the object, using [Segment-Anything](https://github.com/facebookresearch/segment-anything)

 ![Box on the object](/Output/bagpack/mask.png)



 # Example 1

```
python run.py --image ./examples/bagpack.jpg  --class bagpack --output ./Output/bagpack_mask.png 
```
### ORIGINAL
![bagpack.jpg](/examples/bagpack.jpg) 

### Mask
![bagpack_mask.jpg](/Output/bagpack_mask.png)

 # Example 2

```
python run.py --image ./examples/wall_hanging.jpg  --class wall_haning --output ./Output/Wall_Hanging.png 
```
### ORIGINAL
![Wall_Hanging.jpg](/examples/wall_hanging.jpg) 

### Mask
![Wall_Hanging.jpg](/Output/wall_hanging.png)


 # Example 3

```
python run.py --image ./examples/stool.jpeg  --class stool--output ./Output/stool.png 
```

### ORIGINAL
![Wall_Hanging.jpg](/examples/stool.jpeg) 

### Mask
![Wall_Hanging.jpg](/Output/stool_mask.png )


# TASK 2
## Change the position of the segmented object using user prompts

Extract the object from image using mask

![object](Output/task2laptop/object.png)

Shift the object and mask 

![shifted object](Output/task2laptop/shifted_object.png)

![shifted mask](Output/task2laptop/shifted_mask.png)


Create a Black Hole on the image on the position of the shifted object

![Black Hole](Output/task2laptop/black_hole.png)


Paste the shifted object on the Black hole Image

![shifted Object](Output/task2laptop/shifted.png)





## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
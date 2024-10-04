# PixShift




### Prerequisites
```
Python
Wget
```
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






# TASK 1

## Segmentating the Object using text prompt

Using GroundingDino, we are making a Box around the object,which is mentioned in the task, then using Segement-Anything(SAM) we segment the object inside the box. SAM gives us mask image of the object and then we create red mask on the object using mask and original image.

 PROMPT = BAGPACK  

 Detect the object based on text prompt using [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 
 ![Box on the object](/Output/annotated/['bagpack'].jpg)

 Masking of the object, using [Segment-Anything](https://github.com/facebookresearch/segment-anything)

 ![Box on the object](/Output/bagpack/mask.png)

 ### Command for TASK1
```
python run.py --image ./example.jpg --class shelf --output ./generated.png
```

 # Examples

```
python run.py --image ./examples/bagpack.jpg  --class bagpack --output ./Output/bagpack_mask.png 
```
### ORIGINAL
![bagpack.jpg](/examples/bagpack.jpg) 

### Mask
![bagpack_mask.jpg](/Output/bagpack_mask.png)


```
python run.py --image ./examples/wall_hanging.jpg  --class wall_haning --output ./Output/Wall_Hanging.png 
```
### ORIGINAL
![Wall_Hanging.jpg](/examples/wall_hanging.jpg) 

### Mask
![Wall_Hanging.jpg](/Output/Wall_hanging.png)


## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
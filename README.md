# PixShift


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


 PROMPT = BAGPACK  
 Detect the object based on text prompt using GroundingDino  
 ![Box on the object](/Output/annotated/['bagpack'].jpg)

 Masking of the object  
 
 ![Box on the object](/Output/bagpack/mask.png)


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
![Wall_Hanging.jpg](/Output/wall_hanging.png)


## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)

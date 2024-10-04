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
![Wall_Hanging.jpg](/Output/Wall_Hanging.png)

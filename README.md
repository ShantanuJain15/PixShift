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


## Instance Segmentation With Text Prompts
Segmentation using text prompt

```
python run.py --image ./examples/bagpack.jpg  --class bagpack --output ./Output/bagpack_mask.png 

![bagpack.jpg](/examples/bagpack.jpg)

![bagpack_mask.jpg](/Output/bagpack_mask.png)

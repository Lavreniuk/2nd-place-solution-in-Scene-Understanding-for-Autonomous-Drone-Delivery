![Banner image](https://images.aicrowd.com/raw_images/challenges/banner_file/1104/1a3fba52db69aff8375e.png)
# Scene Understanding for Autonomous Drone Delivery (SUADD)

It is [2nd place solution of the seg-dep team](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/leaderboards) (2nd place in Mono-depth estimation and 3rd place in Semantic segmentation) in Scene Understanding for Autonomous Drone Delivery (SUADD'23) competition hosted by Amazon Prime Air and AICrowd.

This is the largest dataset with full semantic annotations and monodepth estimation ground-truth over a wide range of AGLs and different scenes.
This dataset contains birdseye-view greyscale images taken between 5 m and 25 m AGL. Annotations for the semantic segmentation task are fully labelled images across 16 distinct classes, while annotations for the mono-depth estimation task have been computed with geometric stereo-depth algorithms. 

For Semantic segmentation we used model soup with [Mask2Former+ViT-Adapter-L](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) pretrained on ADE20K.   
For Mono-depth estimation we used model soup with [ZoeDepth](https://github.com/isl-org/ZoeDepth) pretrained on NYU Depth v2.

## News
- `2023/09/19`: 🚀🚀 We have presented our solution on [DAGM German Conference on Pattern Recognition](https://edellano.github.io/suadd_workshop/program.html)
- `2023/07/18`: 🚀 Developer of [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main#awesome-competition-solutions-with-vit-adapter) added information about our winning solution using ViT-Adapter

## About the Scene Understanding for Autonomous Drone Delivery Challenge

Unmanned Aircraft Systems (UAS) have various applications, such as environmental  studies, emergency responses or package delivery. The safe operation of fully autonomous  UAS requires robust perception systems. 

For this challenge, we will focus on images of a single downward camera to estimate the scene's depth and perform semantic segmentation. The results of these two tasks can help the development of safe and reliable autonomous control systems for aircraft. 

This challenge includes the release of a new dataset of drone images that will benchmark semantic segmentation and mono-depth perception. The images in this dataset comprise realistic backyard scenarios of variable content and have been taken on various Above Ground Level (AGL) ranges.

## Semantic Segmentation 
### SOTA Segmentation models
* [https://github.com/czczup/ViT-Adapter/tree/main/segmentation](https://github.com/czczup/ViT-Adapter/tree/main/segmentation)
* [https://github.com/baaivision/EVA/tree/master/EVA-01/seg](https://github.com/baaivision/EVA/tree/master/EVA-01/seg)
* [https://github.com/OpenGVLab/InternImage/blob/master/README_EN.md](https://github.com/OpenGVLab/InternImage/blob/master/README_EN.md)

### Usage instructions
```
* git clone https://github.com/czczup/ViT-Adapter/tree/main
* cd ViT-Adapter/segmentation
* pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
* pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
* pip install timm==0.4.12
* pip install mmdet==2.22.0 # for Mask2Former
* pip install mmsegmentation==0.20.2
* ln -s ../detection/ops ./
* cd ops & sh make.sh # compile deformable attention
```

Then use the config provided [here](https://github.com/niveditarufus/suadd/blob/main/mask2former_beitv2_adapter_large_896_80k_ade20k_ss_tr_896.py) to train the model. Our final submission was a model soup of the 18kth and the 20kth iterations models.

## Mono-depth estimation
### SOTA Mono-depth models
* [https://github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)
* [https://github.com/wl-zhao/VPD](https://github.com/wl-zhao/VPD)

### About the Mono Depth Estimation Task

Depth estimation measures the distance between the camera and the objects in the scene.  It is an important perception task for an autonomous aerial drone. Using two stereo cameras makes this task solvable with stereo vision methods. This challenge aims to  create a model that can use the information of a single camera to predict the depth of every pixel. 

The output of this task must be an image of equal size to the input image, in which every pixel contains a depth value.

### Evaluation

Models submitted to the Depth Estimation task will be evaluated accoring to the Scale invariant logarithmic error (SILog) and Abs Rel score. The submission should generate outputs that are valid depth values, non positive values or invalid values will result in a failed submission.

The exact code used for the calculation of SILog can be found in the `si_log` function in [`local_evaluation.py`](https://gitlab.aicrowd.com/aicrowd/challenges/suadd-2023/suadd-2023-depth-perception-starter-kit/-/blob/master/local_evaluation.py)

### Training

For our final solution, we chose the ZoeDepth model and made several changes to the training pipeline and model configuration. The main changes are as follows:
* Image augmentation
* Image size
* Changing horizontal to vertical flip in TTA
* Removing borders in inference
* Making models soup
* Using ensemble in inference

**Dataset:** we use all provided data for training (train + validation datasets) = 2056 images  
**Env:** To reproduce our solution, you should install the same environment as for the original ZoeDepth model, but fro training use folder ZoeDepth from our git, not original one.  
**Dataset config:**
Change the dataset paths in the my_submission/ZoeDepth/zoedepth/utils/config.py in the Lines 105, 106, 110, 111.  
**Training:**
You should train four models using the following configuration files:
* zoedepth\models\zoedepth\config_zoedepth.json
* zoedepth\models\zoedepth\config_zoedepth_2.json
* zoedepth\models\zoedepth\config_zoedepth_3.json
* zoedepth\models\zoedepth\config_zoedepth_4.json

Download pretrained weights:
from my_submission/ZoeDepth run  
`wget https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt`  
for training run  
`CUDA_VISIBLE_DEVICES=0 python train_mono.py -m zoedepth --pretrained_resource=local::ZoeD_M12_N.pt`  
it will train for config_zoedepth.json for other 3 configs do same, but just rename config_zoedepth2.json -> config_zoedepth.json and etc.

**Soup:**
Weights after training could be found here  
`~/shortcuts/monodepth3_checkpoints/`  
Copy only ***_latest.pt** for the first two models (with 896x592 resolution) in some new empty folder and run soup.py, after that do the same for other 2 weighs (with 768x512 resolution). Put them in my_submission\models\dpt_large_896.pt and my_submission\models\dpt_large_768.pt, respectively.

Example:  
```
cd ~/shortcuts/monodepth3_checkpoints/
mkdir soup
cp ZoeDepthv1_03-Jun_19-06-9099dc26351e_latest.pt ZoeDepthv1_04-Jun_12-15-64b1181cc0db_latest.pt soup/
cd /mnt/data/nick/suadd-2023-semantic-segmentation-starter-kit/
python soup.py ~/shortcuts/monodepth3_checkpoints/soup/
mv ~/shortcuts/monodepth3_checkpoints/soup/res.pt my_submission/models/dpt_large_896.pt
rm -rf ~/shortcuts/monodepth3_checkpoints/soup
```

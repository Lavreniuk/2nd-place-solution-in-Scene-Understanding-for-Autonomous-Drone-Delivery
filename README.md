# SUADD

It is [2nd place solution of the seg-dep team](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/leaderboards) (2nd place in Mono-depth estimation and 3rd place in Semantic segmentation) in Scene Understanding for Autonomous Drone Delivery (SUADD'23) competition hosted by Amazon Prime Air and AICrowd.

This is the largest dataset with full semantic annotations and monodepth estimation ground-truth over a wide range of AGLs and different scenes.
This dataset contains birdseye-view greyscale images taken between 5 m and 25 m AGL. Annotations for the semantic segmentation task are fully labelled images across 16 distinct classes, while annotations for the mono-depth estimation task have been computed with geometric stereo-depth algorithms. 

For Semantic segmentation we used model soup with [Mask2Former+ViT-Adapter-L](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) pretrained on ADE20K.   
For Mono-depth estimation we used model soup with [ZoeDepth](https://github.com/isl-org/ZoeDepth) pretrained on NYU Depth v2.

### SOTA Segmentation models
* [https://github.com/czczup/ViT-Adapter/tree/main/segmentation](https://github.com/czczup/ViT-Adapter/tree/main/segmentation)
* [https://github.com/baaivision/EVA/tree/master/EVA-01/seg](https://github.com/baaivision/EVA/tree/master/EVA-01/seg)
* [https://github.com/OpenGVLab/InternImage/blob/master/README_EN.md](https://github.com/OpenGVLab/InternImage/blob/master/README_EN.md)

### SOTA Mono-depth models
* [https://github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)
* [https://github.com/wl-zhao/VPD](https://github.com/wl-zhao/VPD)

### Additional Datasets
* [https://www.tugraz.at/index.php?id=22387](https://www.tugraz.at/index.php?id=22387)
* [https://uavid.nl/](https://uavid.nl/)
* [https://midair.ulg.ac.be/download.html](https://midair.ulg.ac.be/download.html)
* [https://github.com/montefiore-ai/midair-dataset/](https://github.com/montefiore-ai/midair-dataset/)

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

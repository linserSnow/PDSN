# Occlusion Robust Face Recognition based on Mask Learning with Pairwise Differential Siamese Network
## [Arxiv (ICCV2019 Poster)](http://openaccess.thecvf.com/content_ICCV_2019/papers/Song_Occlusion_Robust_Face_Recognition_Based_on_Mask_Learning_With_Pairwise_ICCV_2019_paper.pdf)
## Introduction
This is code for the PDSN in our paper.
## Abstract
Deep Convolutional Neural Networks (CNNs) have been pushing the frontier of face recognition over past years. However, existing general CNN face models generalize poorly for occlusions on variable facial areas. Inspired by the fact that the human visual system explicitly ignores the occlusion and only focuses on the non-occluded facial areas, we propose a mask learning strategy to find and discard corrupted feature elements from recognition. A mask dictionary is firstly established by exploiting the differences between the top conv features of occluded and occlusionfree face pairs using innovatively designed pairwise differential siamese network (PDSN). Each item of this dictionary captures the correspondence between occluded facial areas and corrupted feature elements, which is named Feature Discarding Mask (FDM). When dealing with a face image with random partial occlusions, we generate its FDM by combining relevant dictionary items and then multiply it with the original features to eliminate those corrupted feature elements from recognition. Comprehensive experiments on both synthesized and realistic occluded face datasets show that the proposed algorithm significantly outperforms the state-of-the-art systems.

![](https://github.com/linserSnow/PDSN/blob/master/images/framework.jpg)

![](https://github.com/linserSnow/PDSN/blob/master/images/PDSN_new.jpg)
## Dependencies
- Python 3.6.5
- PyTorch 1.0.0
## Training
### Data preparation
***To be continued...***
### Train one PDSN
run *./scripts/train_dict.sh* with proper settings.
## Construct dictionary
### Data preparation
***To be continued...***
### Extract masks
run *./scripts/extract_mask_dic.sh* with proper settings.
## Contributors
If you find this repository useful for your research, please cite the following paper:

```
  @InProceedings{Song_2019_ICCV,
    author = {Song, Lingxue and Gong, Dihong and Li, Zhifeng and Liu, Changsong and Liu, Wei},
    title = {Occlusion Robust Face Recognition Based on Mask Learning With Pairwise Differential Siamese Network},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

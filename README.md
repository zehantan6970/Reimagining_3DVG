# Reimagining_3DVG

Re_3DVG proposes a method for fragmented point cloud scenarios. Utilizing instance segmentation and transformer models, our approach offers a potent mechanism for establishing robust correspondences between text queries and object instances within the shared visible range.

## Download

The fragmented 3DVG dataset can be downloaded from:

Google Drive

The test dataset can be downloaded from:

Google Drive

The weight module files can be downloaded from:

Google Drive

We also provide weight files in Chinese from：

Google Drive

## Installation

Our project has been tested on torch=1.13 cuda=11.7 python=3.7.5

## Train

You can use the following scripts for the train: T_V_en_base_with_F_4L_e4_BERT.py

## Inference

You can use the example scripts demo.py in the demo directory to perform RGB images, depth images, and text descriptions. 

## Comparison Experiment

See the experiment directory.


## Acknowledgement

We appreciate the open-source of the following projects: [MVT-3DVG](https://github.com/sega-hsj/MVT-3DVG) ,  [OFA](https://github.com/OFA-Sys), [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [ScanRefer](https://github.com/daveredrum/ScanRefer), [EDA](https://github.com/yanmin-wu/EDA ),  [Mask_RCNN](https://github.com/matterport/Mask_RCNN ),
and [ScanNet](https://github.com/ScanNet/ScanNet).

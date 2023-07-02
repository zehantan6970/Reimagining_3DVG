import numpy as np
import os
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2 as cv
import matplotlib.pyplot as plt
# 指定模型的配置文件和 checkpoint 文件路径
# envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# config_file = '/home/light/gree/slam/D3VG/mmdetection_master/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py'
# checkpoint_file = '/home/light/gree/slam/D3VG/mmdetection_master/checkpoint/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth'
# config_file = '/home/light/gree/slam/D3VG/mmdetection_master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_a.py'
# checkpoint_file = '/media/light/light_t2/checkpoints/mask_rcnn/epoch_39.pth'
config_file = 'mmdetection_master/configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco-trash_a.py'
checkpoint_file = 'weights/epoch_62.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')


def generate_seg(img, output_dir=None):
    """
    Args:
        img: 输入的图片
        output_dir: 分割结果保存文件夹路径

    Returns:
        boxes_class:目标框类别,segs:mask坐标,boxes:目标框坐标
    """
    # box类别
    boxes_class = []
    # mask掩模图
    segs = []
    # box
    boxes = []
    # 根据配置文件和 checkpoint 文件构建模型
    # 测试单张图片并展示结果
    result = inference_detector(model, img)
    # 或者将可视化结果保存为图片
    model.show_result(img, result, score_thr=0.5, out_file=output_dir, show=False)
    score = model.BOXES[:, -1]
    choose_index = np.where(score > 0.5)[0]
    label_id = model.LABEL[score > 0.5]
    for i, l in enumerate(label_id):
        if model.CLASSES[l] not in ["book", "picture"]:
            boxes_class.append(model.CLASSES[l])
            segs.append(model.SEGS[choose_index[i]])
            boxes.append(model.BOXES[:, :-1][choose_index[i]])
    return boxes_class, segs, boxes


if __name__ == "__main__":
    img = '/media/light/light_t2/t2/DATA/scannet_frames_25k/scene0000_00/color/001800.jpg'
    output_dir = "result.jpg"
    generate_seg(img, output_dir)

_base_ = [
    './mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco.py'
]

model = dict(
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    roi_head=dict(
        bbox_head=dict(num_classes=38),
        mask_head=dict(num_classes=38)))
dataset_type = 'COCODataset'
classes = ('_background_',"cabinet","bed","chair","sofa","table","door", "window", "bookshelf","picture","counter","blinds","desk","shelves",
            "curtain", "dresser","pillow","mirror","floor mat", "clothes","books","refridgerator","television","paper","towel",
             "shower curtain","box", "whiteboard", "person","night stand","toilet", "sink","lamp","bathtub","bag","otherstructure","otherfurniture","otherprop")

_base_ = './mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=38),
        mask_head=dict(num_classes=38)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('_background_',"cabinet","bed","chair","sofa","table","door", "window", "bookshelf","picture","counter","blinds","desk","shelves",
        "curtain", "dresser","pillow","mirror","floor mat", "clothes","books","refridgerator","television","paper","towel",
        "shower curtain","box", "whiteboard", "person","night stand","toilet", "sink","lamp","bathtub","bag","otherstructure","otherfurniture","otherprop")

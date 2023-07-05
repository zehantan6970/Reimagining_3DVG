from models.model_vit_en_base_patch16_224_in21k_with_finetune_BERT import Model
import torch
from utils.dataloader_vit_base_patch16_224_in21k_with_vit_finetune import dataload,dataload_for_eval
from utils.utils import getRt,transform,get_obb,get_box3d_min_max,get_iou_obb,class_filter,uv2xyz,generate_scannet_seg
import numpy as np
import argparse
import clip
from PIL import Image
from mmdetection_master.mmpredict import generate_seg
import os
import cv2 as cv
# from Segment_anything.Seg_anything import seg_predict
envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# -------- #
# 配置
# -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim',default=512)
parser.add_argument('--dropout',default=0.1)
parser.add_argument('--nheads',default=4)
parser.add_argument('--dim_feedforward',default=2048)
parser.add_argument("--enc_layers",default=4)
parser.add_argument("--dec_layers",default=4)
parser.add_argument("--max_length",default=15)
parser.add_argument("--max_words_length",default=25)
parser.add_argument("--cls_num",default=15)
parser.add_argument("--pre_norm",default=True)
args=parser.parse_args()
vqa_model = Model(args)
# 需要说明是否模型测试
vqa_model.eval()
# 加载模型
train_vqa_para = torch.load("/media/light/light_t2/消融试验对比/T_V_en_base_with_F_4L_e4_BERT/pytorch_model_74_loss_0.010146.bin")
vqa_model.load_state_dict(train_vqa_para)
total = sum([param.nelement() for param in vqa_model.parameters()])
print("可训练参数总量:", total)
def uv2xyz(camera_inter,scalingFactor,depth,uv):
    fx ,fy,centerX,centerY=camera_inter
    # -----------------------------------------------------------------
    # 像素坐标u，v
    # -----------------------------------------------------------------
    u,v=uv
    # -----------------------------------------------------------------
    # 相机坐标X，Y，Z
    # -----------------------------------------------------------------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0:
        return False
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X,Y,Z]
def get_seg_info(rgb_path, depth_path, camera_inter, scale_factor):
    """

    Args:
        rgb_dir: 待分割的image文件夹路径
        depth_path: 深度图文件夹路径
        camera_inter: 相机内参
        scale_factor: 缩放因子

    Returns:
        dict: dtype=dict key: 相机id value: 第一个列表保存的segs分类,第二个列表保存mask在相机坐标系下的坐标,相机坐标系不是数组坐标系，好像是像素坐标系，第三个列表放入seg的box坐标
    """
    # 保存全部rgb的分割结果
    dict = {}
    # ---------------------------------------------------------------------
    # 第一个[]放入seg类别，第二个[]放入seg的mask坐标,第三个[]放入seg的box坐标
    # ---------------------------------------------------------------------
    dict["info"] = [[], [], []]
    boxes_class, segs, boxes = generate_seg(rgb_path,"res.jpg")
    depth = Image.open(depth_path)
    for i, box_class in enumerate(boxes_class):
        # where返回的是numpy坐标，numpy坐标转化为像素坐标需要调换
        v, u = np.where(segs[i] == True)
        # u = np.array((u / (1296 / 640) - 1),dtype=np.int32)[::4]
        # v = np.array((v / (968 / 480) - 1),dtype=np.int32)[::4]
        u = np.array((u ), dtype=np.int32)
        v = np.array((v), dtype=np.int32)
        uv = np.concatenate([u.reshape([u.shape[0], 1]), v.reshape([v.shape[0], 1])], axis=1)
        if len(uv)>500:
            dict["info"][0].append(box_class)
            dict["info"][2].append(boxes[i])
            xyz_list = []
            for uv_ in uv:
                # ---------------------------------------------------------------------
                # 第二步 把mask的uv坐标转换为相机坐标系的xyz坐标
                # ---------------------------------------------------------------------
                xyz = uv2xyz(camera_inter, scale_factor, depth, uv_)
                if xyz:
                    xyz_list.append(xyz)
            dict["info"][1].append(xyz_list)
    return dict
def data_info_extractor(scannet_frames_25k, scan_frame_id):
    CAMERA_INTER = [577.5, 578.7, 318.9, 242.7]
    # scaling factor
    SCALE_FACTOR = 1000
    scene_id=scan_frame_id[:4]
    frame_id=scan_frame_id[-6:]
    an_id=scan_frame_id[4:6]
    # color_img=os.path.join(scannet_frames_25k,"scene{}_{}".format(scene_id,an_id),"color","{}.jpg".format(frame_id))
    # depth_img = os.path.join(scannet_frames_25k, "scene{}_{}".format(scene_id,an_id), "depth", "{}.png".format(frame_id))
    color_img = os.path.join("/home/light/0614/image-0614-1/rgb/1686724401.867864.png")
    depth_img = os.path.join("/home/light/0614/image-0614-1/depth/1686724401.867864.png")
    t1=time.time()
    dict_info=get_seg_info(color_img,depth_img,CAMERA_INTER,SCALE_FACTOR)
    t2=time.time()
    # print("消耗时间：%f"%(t2-t1))
    region_lst,center_lst,d3boxes=get_proposals(dict_info, color_img)
    return region_lst,center_lst,d3boxes
def get_proposals(dict_info, color_img):
    boxes=dict_info["info"][2]
    seg_points=dict_info["info"][1]
    # ----------------------- #
    # 截取region
    # ----------------------- #
    region_lst=[]
    center_lst=[]
    d3boxes=[]
    image=Image.open(color_img)
    for i,box in enumerate(boxes):
        region = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        region.save("/home/light/文档/真实场景测试/region/{}.jpg".format(i))
        center, vertex_set = get_obb(seg_points[i])
        if center!=None:
            region_lst.append(region)
            center_lst.append(center)
            d3boxes.append(vertex_set)
    print(center_lst)
    return region_lst,center_lst,d3boxes
import time
def cal_iou(region_lst,center_lst,d3boxes,eval_question,scene):
    # 提取regions特征
    regions_feature=dataload_for_eval(region_lst)
    test_patch = [regions_feature]
    # regions中心坐标
    test_d3_patches = [center_lst]
    # 问题
    test_question = [eval_question[0]]
    Id=eval_question[1]
    t1 = time.time()
    logits = vqa_model(test_patch, np.array(test_d3_patches) / np.max(abs(np.array(test_d3_patches))),
                       test_question)
    print(torch.nn.functional.softmax(logits,dim=1))
    t2 = time.time()
    print(t2 - t1)
    # print("预测的结果为:",torch.argmax(logits, dim=1))
    # print("该物体的3dbox坐标为：",d3boxes[int(torch.argmax(logits, dim=1).cpu().detach().numpy())])
    # 获取预测的box的坐标
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1=get_box3d_min_max(d3boxes[int(torch.argmax(logits, dim=1).cpu().detach().numpy())])
    # 显示预测的物体
    region_lst[int(torch.argmax(logits, dim=1).cpu().detach().numpy())].show()
    region_lst[int(torch.argmax(logits, dim=1).cpu().detach().numpy())].save("/home/light/gree/slam/D3VG/temperary_dir_4L/{}_{}.jpg".format(scene,test_question))
    # 真实的物体的box的坐标
    arr = np.load("/media/light/light_t2/t2/DATA/my_scan/scannet_v4/scans/{}/npy/{}_bbox.npy".format(scene,scene))
    x_min_2,y_min_2,z_min_2,x_max_2,y_max_2,z_max_2,label_id,obj_id= arr[Id]
    # 计算iou
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)
    print("iou",iou)
    return iou


# with open("/media/light/light_t2/t2/DATA/my_scan/scannet_v4/scanrefer_en.txt",mode="r") as r:
#     lines=r.readlines()
#     s=""
#     num=0
#     num_05=0
#     t1=time.time()
#     for l in lines:
#
#         scene=l.split("|")[0]
#         question=l.split("|")[1]
#         Id=int(l.split("|")[2])
#         if scene!=s:
#             # 提取分割模型分割后的参数
#             region_lst, center_lst, d3boxes = data_info_extractor("/media/light/light_t2/t2/DATA/scannet_frames_test",
#                                                                   scene[5:9]+scene[10:12]+scene[14:])
#             iou=cal_iou(region_lst, center_lst, d3boxes, [question, Id],scene)
#             s=scene
#             if iou>0.25:
#                 num+=1
#             if iou>0.5:
#                 num_05+=1
#         else:
#             iou=cal_iou(region_lst, center_lst, d3boxes, [question, Id],scene)
#             if iou>0.25:
#                 num+=1
#             if iou > 0.5:
#                 num_05 += 1
#     t2=time.time()
#     print("消耗总时间为:%fs"%(t2-t1))
#     print(num)
#     print(num_05)
# # dirs=os.listdir("/media/light/light_t2/t2/DATA/my_scan/scannet_v4/scans")
# # all_time=0
# # for scene in dirs:
# #     t1=time.time()
# #     region_lst, center_lst, d3boxes = data_info_extractor("/media/light/light_t2/t2/DATA/scannet_frames_test",
# #                                                           "{}".format(scene[5:9]+scene[10:12]+scene[14:]))
# #     iou=cal_iou(region_lst, center_lst, d3boxes, ["The behind chair", 1],"{}".format(scene))
# #     t2=time.time()
# #     all_time+=(t2-t1)
# #     print(t2-t1)
# # print(all_time)
region_lst, center_lst, d3boxes = data_info_extractor("/media/light/light_t2/t2/DATA/scannet_frames_test",
                                                      "070700000000")

iou=cal_iou(region_lst, center_lst, d3boxes, ["The right chair", 1],"{}".format("scene0707_00_000000"))
#

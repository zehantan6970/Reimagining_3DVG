from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2 as cv
import os
import numpy as np
from PIL import Image
envpath = '/home/light/anaconda3/envs/ofa/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
import time
ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_distilled_en')
import os
import json
scans_refer_path="/media/light/light_t2/t2/DATA/my_scan/scannet_v4/val.json"
# with open(scans_refer_path,"r") as r:
#     data=json.load(r)
# for da in data:
#     scene_id=da["scene_id"]
#     description=da["description"]
#     print(description)
#     image = "/media/light/light_t2/t2/DATA/scannet_frames_test/{}/color/{}.jpg".format(scene_id[:12],scene_id[13:])
#     input = {'image': image, 'text': description}
#     result = ofa_pipe(input)
#     cv_image=cv.imread(image)
#     box=result[OutputKeys.BOXES][0]
#     print(result[OutputKeys.BOXES][0])
#     draw_image= cv.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))#cv2.rectangle(image, pt1,pt2, color)
#
#     image = Image.fromarray(cv.cvtColor(draw_image, cv.COLOR_BGR2RGB))
#     image.show()
#     time.sleep(5)
t1=time.time()
image = "/media/light/light_t2/t2/DATA/scannet_frames_test/scene0712_00/color/001000.jpg"
description="black backpack"
input = {'image': image, 'text': description}
result = ofa_pipe(input)
cv_image=cv.imread(image)
box=result[OutputKeys.BOXES][0]
print(result[OutputKeys.BOXES][0])
draw_image= cv.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))#cv2.rectangle(image, pt1,pt2, color)
image = Image.fromarray(cv.cvtColor(draw_image, cv.COLOR_BGR2RGB))
image.show()
t2=time.time()
print(t2-t1)

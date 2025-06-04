import os
import cv2
import re
import random
import numpy as np
import matplotlib.pyplot as plt


# make hazy data
def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def bgr_to_rgb(img):
    b, g, r = cv2.split(img)
    img_rgb = cv2.merge([r, g, b])
    return img_rgb


def get_haze(img, depth_img):

    depth_img = 255 - depth_img * 255

    depth_img_3c = np.zeros_like(img)
    depth_img_3c[:, :, 0] = depth_img
    depth_img_3c[:, :, 1] = depth_img
    depth_img_3c[:, :, 2] = depth_img

    beta = random.randint(100, 300) / 100
    norm_depth_img = depth_img_3c / 255
    trans = np.exp(-norm_depth_img * beta)

    A = 255
    # A = random.randint(100, 255)
    hazy = img * trans + A * (1 - trans)
    hazy = np.array(hazy, dtype=np.uint8)

    return hazy


# filepath = r'./input/'  # 图片所在的文件夹
filepath = 'D:/Environment/paper/jxk-code/YOLO/data/mydata/origin/test/images/'  # 图片所在的文件夹
pathDir = os.listdir(filepath)


for i, allDir in enumerate(pathDir):
    # imgpath = r'./input/' + allDir
    imgpath = 'D:/Environment/paper/jxk-code/YOLO/data/mydata/origin/test/images/' + allDir
    # dep_imgpath = r'./output-d/' + allDir
    dep_imgpath = 'D:/Environment/paper/jxk-code/YOLO/data/mydata/origin/test/output-d/' + allDir

    dep_imgpath = re.sub('jpg', 'png', dep_imgpath)

    img = plt.imread(imgpath)
    depth_img = plt.imread(dep_imgpath)

    hazy = get_haze(img, depth_img)

    cv2.imwrite('D:/Environment/paper/jxk-code/YOLO/data/mydata/origin/test/output-fog/' + allDir, bgr_to_rgb(hazy))
    print('Finished fogging:'+allDir+'(%g/%g)' % (i+1, len(pathDir)))

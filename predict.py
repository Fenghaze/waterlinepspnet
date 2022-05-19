'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要原图和分割图不混合，可以把blend参数设置成False。
4、如果想根据mask获取对应的区域，可以参考detect_image中，利用预测结果绘图的部分。
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(self.num_classes):
    seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
'''
import time

import numpy
from tqdm import tqdm

from pspnet import PSPNet
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from post_process import *

pspnet = PSPNet(blend=False)

# 测试图片文件夹
def pred_file(imgs_path, save_mask=False, fitline=True):
    total_time = 0
    imgs = os.listdir(imgs_path)
    for img in imgs:
        img_name = img.split('.jpg')[0]
        img_path = os.path.join(imgs_path, img)
        input = Image.open(img_path)
        input = input.convert('RGBA')
        s_time = time.time()
        pred, final = pspnet.detect_image(input)

        ######## 预测mask ########
        if save_mask:
            cv2.imwrite(save_path + str(img_name) + '.png', pred)
            cv2.imwrite(save_path + str(img_name) + '_mask.png', final)
        #########################

        ######## 后处理 ########
        raw_img = cv2.cvtColor(numpy.asarray(input), cv2.COLOR_BGR2RGB)
        mask_edge_detection(raw_img, str(img_name), pred, fitline=fitline)
        #######################

        u_time = time.time()
        img_time = u_time - s_time
        total_time += img_time
        #print("image:{} time: {} ".format(img_name, img_time))
    avg_time = total_time / len(imgs)
    return avg_time


# 测试VOC划分的测试集
def pred_voc_test(test_txt, save_mask=False):
    total_time = 0
    image_ids = open(test_txt, 'r').read().splitlines()
    for image_id in image_ids:
        s_time = time.time()
        img_path = r"D:\Deeplearning\data\ships7\JPEGImages" + '/' + image_id + ".jpg"
        input = Image.open(img_path)
        input = input.convert('RGBA')

        ######## 预测mask ########
        pred, final = pspnet.detect_image(input)
        if save_mask:
            cv2.imwrite(save_path + str(image_id) + '.png', pred)
            cv2.imwrite(save_path + str(image_id) + '_mask.png', final)
        #########################

        ######## 后处理 ########
        raw_img = cv2.cvtColor(numpy.asarray(input), cv2.COLOR_BGR2RGB)
        mask_edge_detection(raw_img, str(image_id), pred)
        #######################

        u_time = time.time()
        img_time = u_time - s_time
        print("image:{} time: {} ".format(image_id, img_time))
        total_time += img_time
    avg_time = total_time/len(image_ids)

# 测试单张图片
def test_per_image(img, save_mask=False, fitline=True):
    img_name = img.split('/')[2].split('.jpg')[0]
    s_time = time.time()
    input = Image.open(img)
    input = input.convert('RGBA')

    ######## 预测mask ########
    pred, final = pspnet.detect_image(input)
    if save_mask:
        cv2.imwrite(save_path + str(img_name) + '.png', pred)
        cv2.imwrite(save_path + str(img_name) + '_mask.png', final)
    #########################

    ######## 后处理 ########
    raw_img = cv2.cvtColor(numpy.asarray(input), cv2.COLOR_BGR2RGB)
    mask_edge_detection(raw_img, str(img_name), pred, fitline=fitline)
    #######################

    u_time = time.time()
    img_time = u_time - s_time
    print("image:{} time: {} ".format(img_name, img_time))

# 测试视频，debug
def video():
    # -------------------------------------#
    #   调用摄像头/视频
    #   capture=cv2.VideoCapture("1.mp4")
    # -------------------------------------#
    capture = cv2.VideoCapture(0)

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(pspnet.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)

        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break

if __name__ == '__main__':
    save_path = './results/'
    # imgs_path = r"D:\Deeplearning\data\ship_test\out\raw"
    # avg_time = pred_file(imgs_path)
    # print("speed(sec/img):", avg_time)
    img_path = "./test_img/1.jpg"
    test_per_image(img_path, True, True)
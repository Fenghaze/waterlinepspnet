import cv2 as cv
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import cv2

def contours_info(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

# 修改目标的像素颜色
def edit_pixel_color(img_path):
    img_lst = os.listdir(img_path)
    for img_id in img_lst:
        img_name = os.path.join(img_path, img_id)
        i = 1
        j = 1
        img = Image.open(img_name)  # 读取系统的内照片
        width = img.size[0]  # 长度
        height = img.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if (data[0]==128):  # 判断RGBA的R值
                    # 判断条件就是一个像素范围范围
                    img.putpixel((i,j),(0,0,0))#则这些像素点的颜色改成黑色
        img = img.convert("RGB")  # 把图片强制转成RGB
        img.save(img_id)  # 保存修改像素点后的图片

# Hu矩匹配相似度
def hu_matrix_compare(gt_img, pre_img, img_name):
    src = cv.imread(gt_img)
    src2 = cv.imread(pre_img)

    # 轮廓发现
    contours1 = contours_info(src)
    contours2 = contours_info(src2)

    # 几何矩计算与hu矩计算
    mm2 = cv.moments(contours2[0])
    hum2 = cv.HuMoments(mm2)

    # 轮廓匹配
    for c in range(len(contours1)):
        mm = cv.moments(contours1[c])
        hum = cv.HuMoments(mm)
        dist = cv.matchShapes(hum, hum2, cv.CONTOURS_MATCH_I1, 0)
        if dist < 1:
            cv.drawContours(src, contours1, c, (0, 0, 255), 2, 8)
        print("dist %f" % (dist))
    #cv.imwrite(img_name, src)
    return dist

# 单通道直方图相似度
def histogram_compare(gt_img, pre_img, img_id):
    # 利用python实现实现图像识别:比较两图像直方图相似性
    # 核心：计算直方图重合度
    image1 = cv2.imread(gt_img, cv2.COLOR_RGB2GRAY)
    image2 = cv2.imread(pre_img, cv2.COLOR_RGB2GRAY)
    # 最简单的以灰度直方图作为相似比较的实现
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist1 = cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX, -1)

    # 可以比较下直方图
    plt.plot(range(256), hist1, 'r')
    plt.plot(range(256), hist2, 'b')
    #plt.show()
    #plt.savefig(img_id+'.png', bbox_inches='tight')

    match1 = cv.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    plt.title('histogram_degree=%.2f\nBHATTA=%.2f' % (degree, match1), fontsize=18)
    plt.savefig(img_id, bbox_inches='tight')
    plt.show()
    print("degree %f" % (degree))
    return degree, match1

# 差值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result

# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def hash_compare(gt_img, pre_img):
    img1 = cv2.imread(gt_img)
    img2 = cv2.imread(pre_img)

    hash1 = dhash(img1)
    #print('img1的dhash值', hash1)
    hash2 = dhash(img2)
    #print('img2的dhash值', hash2)
    camphash2 = campHash(hash1, hash2)
    #print("dhash差异哈希相似度：", camphash2)
    return camphash2

if __name__ == '__main__':
    # img_path = r'D:\Deeplearning\data\ship_test\out\label_waterline'
    # edit_pixel_color(img_path)

    #### 计算绘制了吃水线的RGB图像的直方图，值越大，越相似 ####
    #### 计算绘制了吃水线的RGB图像的差值哈希，值越小，越相似 ####
    gt_path = r'D:\Deeplearning\data\ship_test\out\raw-waterline'
    pre_path = r'D:\Deeplearning\data\ship_test\out\pre-waterline'
    gt_lst = os.listdir(gt_path)
    pre_lst = os.listdir(pre_path)
    historgram = 0
    hash = 0
    match = 0
    hu = 0
    img_len = len(gt_lst)
    for gt in gt_lst:
        gt_img = os.path.join(gt_path, gt)
        pre = gt.split('.jpg')[0] + '_fitline3.png'
        pre_img = os.path.join(pre_path, pre)
        historgram1, match1= histogram_compare(gt_img, pre_img, gt)
        historgram += historgram1
        match += match1
        #hash += hash_compare(gt_img, pre_img)

    print("historaram = ", historgram/img_len)
    print("match = ", match/img_len)

    #print("hash = ", hash/img_len)

    #### 计算绘制了吃水线的二值图的Hu矩匹配程度，值越小，越相似 ####
    gt_path = r'D:\Deeplearning\data\ship_test\out\blank_gt_waterline'
    pre_path = r'D:\Deeplearning\data\ship_test\out\blank_pre_waterline'
    gt_lst = os.listdir(gt_path)
    pre_lst = os.listdir(pre_path)
    for gt in gt_lst:
        gt_img = os.path.join(gt_path, gt)
        pre = gt.split('.png')[0] + '_fitline3.png'
        pre_img = os.path.join(pre_path, pre)
        hu += hu_matrix_compare(gt_img, pre_img, gt)

    print("hu = ", hu/img_len)
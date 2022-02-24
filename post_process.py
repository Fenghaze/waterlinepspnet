# -*- encoding: utf-8 -*-
"""
@File    : post_process.py
@Time    : 2021/4/8 11:11
@Author  : Zhl
@Desc    : 后处理阶段
"""

import random
import scipy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import copy

save_dir = './results/'

# 使用霍夫直线检测
class Line_of_horizont_fitting:
    line = []

    def inputNormalized(self, image, img_w, img_h):
        image = self.resize_image(image, img_w, img_h)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        return image

    def resize_image(self, image, img_w, img_h):
        image = cv2.resize(image, (img_w, img_h))  # 160x160
        return image

    def plot_binary_image(self, image):
        image = image * 255
        binary_img = np.squeeze(image, axis=2)
        plt.imshow(binary_img)
        plt.show()

    def get_binary_image(self, image, treshold):
        image = cv2.threshold(image, treshold, 1, cv2.THRESH_BINARY)
        return image[1]

    # 提取mask信息
    def binary_edge_detection(self, image):
        edges = image - scipy.ndimage.morphology.binary_dilation(image)
        return edges

    def median_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.blur(img, (kernel_size, kernel_size))

    # 在图像中查找直线
    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
        - image：二值图
        - rho：线段以像素为单位的距离精度，double类型的，推荐用1.0
        - theta：线段以弧度为单位的角度精度，推荐用numpy.pi/180
        - threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
        - lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在
        - minLineLength：线段以像素为单位的最小长度，根据应用场景设置
        - maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        return lines

    def binary2gray(self, image):
        image = np.uint8(255 * image)
        return image

    def Collect_points(self, lines):

        # interpolation & collecting points for RANSAC
        points = []
        if type(lines.tolist()[0]) != list:
            lines = lines.reshape(1, 4)
        for line in lines:
            new_point = np.array([[int(line[0]), int(line[1])]])
            points.append(new_point)
            new_point = np.array([[int(line[2]), int(line[3])]])
            points.append(new_point)

        return points

    def smoothing(self, lines, pre_frame=10):
        # collect frames & print average line
        lines = np.squeeze(lines)
        avg_line = np.array([0.0, 0.0, 0.0, 0.0])

        for ii, line in enumerate(reversed(lines)):
            if ii == pre_frame:
                break
            avg_line += line
        avg_line = avg_line / pre_frame

        return avg_line

    def delete_line(self, lines, h, w):
        #计算lines中每条line的倾角
        res_lines = []
        i = 1
        for line in lines:
            # newlines1 = lines[:, 0, :]
            x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
            #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
            # 转换为浮点数，计算斜率
            x1 = float(x1)
            x2 = float(x2)
            y1 = float(y1)
            y2 = float(y2)
            if x2 - x1 == 0:    #垂直线
                result = 90
            elif y2 - y1 == 0:  # 水平线保留
                if abs(h - y1) >= h * 0.6:
                    continue
                result = 0
                res_lines.append(line)
            elif abs(h - y1) >= h * 0.4:
                continue
            else:
                # 计算斜率
                k = -(y2 - y1) / (x2 - x1)
                # 求反正切，再将得到的弧度转换为度
                result = np.arctan(k) * 57.29577    #直线倾斜角度
                if 0 < abs(result) <= 35:   #倾角保留
                    res_lines.append(line)
            i = i + 1
        if len(res_lines) == 0:
            res_lines.append(np.array([[0, h, w, h]]))
        return res_lines

    def getLineImage(self, image, label, fit_line, width, height):
        height, width, _ = image.shape
        imageOUT = cv2.bitwise_or(image, label)
        cv2.line(imageOUT, (int(fit_line[2] - fit_line[0] * width), int(fit_line[3] - fit_line[1] * width)),
                 (int(fit_line[2] + fit_line[0] * width), int(fit_line[3] + fit_line[1] * width)), (255, 0, 255), 12)

    def predict_segmentation(self, image, model):
        predict = model.predict(image[None, ...])

        return predict[0]

    # 边缘检测得到二值图后，使用霍夫直线检测检测直线，并保存结果
    def detect_line(self, img, edge_segmentation, img_name, kernel_median_blur=50,
                                       predict_treshold=0.5, rho=2, theta=np.pi / 180, threshold=50, min_line_length=15,
                                       max_line_gap=30, fitline=True):
        h, w = edge_segmentation.shape
        # 检测分割图像的边缘线段
        lines = self.hough_lines(edge_segmentation, rho, theta, threshold, min_line_length, max_line_gap)
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
            houghOut = cv2.line(edge_segmentation, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
        save_name = save_dir + img_name + '_houghline.png'
        cv2.imwrite(save_name, houghOut)
        # 去掉垂直、倾角大的线段
        lines = self.delete_line(lines, h, w)


        # 没有检测到吃水线段时，return -1
        if np.sum(lines == None):
            return -1
        line_arr = np.squeeze(lines)
        # 图像中的线段高度由高到低排序
        # line_arr = line_arr[line_arr[:, 1].argsort()][::-1]

        points = self.Collect_points(line_arr)  #线段上的所有点

        if (len(points) < 2):
            points = line_arr.reshape(lines.shape[0] * 2, 2)

        if (len(points) >= 2):
            fit_line = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 1, 0.001, 0.001)
            # self.line.append(fit_line)
            #
            # if len(self.line) > 10:
            #    fit_line = self.smoothing(self.line, 10)

        x0 = (int(fit_line[2] - (w * fit_line[0])))
        x1 = (int(fit_line[2] + (w * fit_line[0])))
        y0 = (int(fit_line[3] - (h * fit_line[1])))
        y1 = (int(fit_line[3] + (h * fit_line[1])))

        x0 = int(fit_line[2] - fit_line[0] * h)
        y0 = int(fit_line[3] - fit_line[1] * w)
        x1 = int(fit_line[2] + fit_line[0] * h)
        y1 = int(fit_line[3] + fit_line[1] * w)
        # 不拟合，直接绘制轮廓线
        if not fitline:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    lineOut = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            save_name = save_dir + img_name + '_line.png'
            cv2.imwrite(save_name, lineOut)
        # 拟合所有直线段
        else:
            # imageOUT = cv2.line(imageOUT, (x0, y0), (x1, y1), (255, 0, 255), 5)
            fitlineOut = cv2.line(img, (x0, y0),(x1, y1),(255, 0, 255), 3)
            save_name = save_dir + img_name + '_fitline.png'
            cv2.imwrite(save_name, fitlineOut)
        # 过滤垂直的线，必须在二值图上进行操作
        # hline = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h/(h*0.4)), 1))
        # imageOUT = cv2.morphologyEx(imageOUT, cv2.MORPH_OPEN, hline)

        #waterline = cv2.cvtColor(imageOUT, cv2.COLOR_GRAY2RGB) # 转为三通道
        #imgadd = cv2.add(img, waterline)

        # save_name = save_dir + img_name + '_fitline.png'
        # cv2.imwrite(save_name, imageOUT)

    def horizont_line_pipeline_verbose(self,img, pred_img, kernel_median_blur=50,
                                       predict_treshold=0.5, rho=2, theta=np.pi / 180, threshold=20, min_line_length=20,
                                       max_line_gap=5):
        h, w, _ = pred_img.shape
        predict_segmentation = pred_img #[h,w,3]
        predict_segmentation = cv2.cvtColor(predict_segmentation, cv2.COLOR_BGR2GRAY)
        predict = self.median_blur(predict_segmentation, kernel_median_blur)
        predict = self.get_binary_image(predict, predict_treshold)
        output = self.binary_edge_detection(predict)    #[h,w,3]
        output = self.binary2gray(output)
        rho = rho  # distance resolution in pixels of the Hough grid
        theta = theta  # angular resolution in radians of the Hough grid
        threshold = threshold  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = min_line_length  # minimum number f pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments
        lines = self.hough_lines(output, rho, theta, threshold, min_line_length, max_line_gap)
        # 没有检测到水岸线时，返回None
        if np.sum(lines == None):
            return None
        line_arr = np.squeeze(lines)
        points = self.Collect_points(line_arr)

        if (len(points) < 2):
            points = line_arr.reshape(lines.shape[0] * 2, 2)

        if (len(points) > 2):
            fit_line = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 1, 0.001, 0.001)
            # self.line.append(fit_line)

            # if len(self.line) > 10:
            #    fit_line = self.smoothing(self.line, average_n_frame)

        pred_visual = predict_segmentation * 255
        #pred_visual = np.uint8(np.concatenate((predict_segmentation, predict_segmentation, pred_visual), axis=2))

        x0 = (int(fit_line[2] - (w * fit_line[0])))
        x1 = (int(fit_line[2] + (w * fit_line[0])))
        y0 = (int(fit_line[3] - (h * fit_line[1])))
        y1 = (int(fit_line[3] + (h * fit_line[1])))
        flag = True
        if flag:
            # imageOUT = cv2.line(imageOUT, (x0, y0), (x1, y1), (255, 0, 255), 5)
            imageOUT = cv2.line(img,
                                (int(fit_line[2] - fit_line[0] * h),
                                 int(fit_line[3] - fit_line[1] * w)),
                                (int(fit_line[2] + fit_line[0] * h),
                                 int(fit_line[3] + fit_line[1] * w)),
                                (255, 0, 255), 5)
        # 不拟合，直接绘制轮廓线
        if not flag:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    imageOUT = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("res", img)
        cv2.waitKey(0)

# 提取mask轮廓，不进行边缘检测
def denoise_mask(img_name, img):
    h, w, _ = img.shape
    blured = cv2.blur(img, (5, 5))  # 进行滤波去掉噪声
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填充
    cv2.floodFill(blured, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)
    # 得到灰度图
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    # 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # 求二值图
    ret, binary = cv2.threshold(closed, 250, 255, cv2.THRESH_BINARY)
    # 找到轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # 绘制结果
    save_name = save_dir + img_name+'_denoise_mask.png'
    cv2.imwrite(save_name, img)

# 1、对分割预测的mask进行边缘检测
def mask_edge_detection(img, img_name, pred_mask, fitline=True):
    """
    :param img: 原图
    :param pred_mask: 分割预测的mask
    """
    fit = Line_of_horizont_fitting()
    h, w, _ = pred_mask.shape

    #方案一：不使用边缘检测，对mask提取外部轮廓
    #denoise_mask(img_name, pred_mask.copy())

    # canny 边缘检测
    pred = cv2.GaussianBlur(pred_mask, (3, 3), 0)
    edge = cv2.Canny(pred, 50, 150) #[h,w]
    save_name = save_dir + img_name +'_edge.png'
    cv2.imwrite(save_name, edge)
    # 方案二：检测边缘检测的直线
    fit.detect_line(img, edge, img_name, fitline=fitline)


    # 方案三：使用边缘检测，提取边缘检测的轮廓
    #extract_contours_fitline(edge, img_name, img)

    ### 将边缘轮廓绘制到原图上 ###
    # canny = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB) # 转为三通道
    # imgadd = cv2.add(img, canny)
    # cv2.imshow('imgadd', imgadd)
    # cv2.waitKey(0)

# 2、边缘检测后去噪：利用 cv2.findContours 找到所有轮廓点，计算轮廓面积，小于面积阈值的轮廓点填充为背景色
def denoise(img, h, w):
    """
    :param img: 边缘检测图像（黑色背景，白色边缘） [h, w]
    """
    gray = img
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    threshold = h / 90 * w / 90 # 面积阈值
    # 提取边缘检测后的所有轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_points = None   # 记录所有轮廓的点集合
    denoise_points = None # 记录去噪后的所有轮廓的点集合

    flag = False
    # 图像中可能包含多条轮廓，contours是一个列表，列表中每个元素表示一条轮廓的点集合
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算当前轮廓所占面积
        if i == 0:
            all_points = contours[i]
        else:
            all_points = np.vstack((all_points, contours[i]))
        if area < threshold:  # 将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv2.drawContours(img, [contours[i]], -1, (0,0,0), thickness=2)  # 原始图片（canny生成的图片）背景BGR值(0,0,0)black
        # 如果当前轮廓未过滤且 all_points 为空，则记录点集
        elif flag == False:
            denoise_points = contours[i]
            flag = True
        # 如果 all_points 已有数据，则将数据垂直拼接
        else:
            denoise_points = np.vstack((denoise_points, contours[i]))

    #save_name = save_dir + 'denoise.png'
    #cv2.imwrite(save_name, img)
    # 返回原始的轮廓点集、去噪后的所有轮廓点集
    return all_points, denoise_points

# 3、获得去噪后的轮廓并拟合吃水线
def extract_contours_fitline(edge_img, img_name, src):
    """
    :param edge_img[h, w]: 边缘检测图像（黑色背景，白色边缘）,注意需要将edge_img转为三通道图像才能drawContours
    :param src: 原图
    """
    height, width = edge_img.shape
    contours = None
    all_contours, denoise_contours = denoise(edge_img, height, width) # 获得去噪后的点集
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)   # 转为三通道图像
    # 未去噪的轮廓边缘
    cv2.drawContours(edge_img, all_contours, -1, (0, 255, 255), 1)  # 吃水线填充为(0, 255, 255)
    # save_name = save_dir + img_name + '_countors.png'
    # cv2.imwrite(save_name, edge_img)
    step_points(all_contours, denoise_contours, edge_img, width)

    # ==========筛选的坐标点个数（较为重要，决定拟合出来的吃水线斜率）=========== #
    # if denoise_contours is None:
    #     contours = all_contours
    #     points_num = 400
    # else:
    #     contours = denoise_contours
    #     points_num = int(len(contours) * 0.1)
    # # ==========筛选的坐标点个数（较为重要，决定拟合出来的吃水线斜率）=========== #
    #
    # points_sort = contours[np.lexsort(-contours[:, 0].T)]   #按 y 值由高到低排序（吃水线最接近图像底部）
    # cv2.drawContours(edge_img, points_sort[:points_num], -1, (0, 255, 255), 1)   # 吃水线填充为(0, 255, 255)
    # cv2.drawContours(edge_img, points_sort[points_num:], -1, (0, 0, 0), 2)   # 其他点集填充为黑色(0, 0, 0)
    #
    # # 拟合吃水线
    # fit_line = cv2.fitLine(points_sort[:points_num], cv2.DIST_L2, 0, 0.01, 0.01)
    # # 绘制吃水线
    # cv2.line(edge_img, (int(fit_line[2] - fit_line[0] * width),
    #                      int(fit_line[3] - fit_line[1] * width)),
    #          (int(fit_line[2] + fit_line[0] * width),
    #           int(fit_line[3] + fit_line[1] * width)), (0, 255, 0), 1)

    # cv2.imshow("img", edge_img)
    # cv2.waitKey(0)
    # save_name = save_dir + img_name + '_fitline.png'
    # cv2.imwrite(save_name, edge_img)

    # 在原图上显示拟合的吃水线
    imgadd = cv2.add(src, edge_img)
    # cv2.imshow('imgadd', imgadd)
    # cv2.waitKey(0)
    save_name = save_dir + img_name + '_waterline.png'
    cv2.imwrite(save_name, imgadd)

# 选择多种像素点进行直线拟合
def step_points(all_contours, denoise_contours, edge_img, width):

    step_points_num = [50, 100, 150, 175, 200, 225, 250, 275, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435] # k=len(step_points_num)，k条吃水线

    # ==========筛选的坐标点个数（较为重要，决定拟合出来的吃水线斜率）=========== #
    if denoise_contours is None:
        contours = all_contours
    else:
        contours = denoise_contours
        points_num = int(len(contours) * 0.1)
    # ==========筛选的坐标点个数（较为重要，决定拟合出来的吃水线斜率）=========== #
    loc = []    #记录k条吃水线的坐标点
    for points_num in step_points_num:
        points_sort = contours[np.lexsort(-contours[:, 0].T)]  # 按 y 值由高到低排序（吃水线最接近图像底部）
        cv2.drawContours(edge_img, points_sort[:points_num], -1, (0, 255, 255), 1)  # 吃水线填充为(0, 255, 255)
        cv2.drawContours(edge_img, points_sort[points_num:], -1, (0, 0, 0), 2)  # 其他点集填充为黑色(0, 0, 0)

        # 拟合吃水线
        fit_line = cv2.fitLine(points_sort[:points_num], cv2.DIST_L2, 0, 0.01, 0.01)
        # 绘制每一条吃水线
        cv2.line(edge_img, (int(fit_line[2] - fit_line[0] * width),
                            int(fit_line[3] - fit_line[1] * width)),
                 (int(fit_line[2] + fit_line[0] * width),
                  int(fit_line[3] + fit_line[1] * width)), (255, 255, 0), 1)
        # 获得直线
        x1 = fit_line[2]
        y1 = fit_line[3]
        loc.append([x1[0], y1[0]])      #直线上的某一个坐标点
        k = fit_line[1] / fit_line[0]   #斜率k
        b = y1 - k * x1                 #偏移量b
        for i in range(random.randrange(10, 100)):     #随机获取直线上的i个坐标点
            x2 = [random.randrange(100, width)]
            y2 = k * x2 + b
            loc.append([x2[0], y2[0]])
    # 绘制吃水线
    loc = np.array(loc)
    # 对k条吃水线上的坐标点进行拟合
    fit_line = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
    cv2.line(edge_img, (int(fit_line[2] - fit_line[0] * width),
                        int(fit_line[3] - fit_line[1] * width)),
             (int(fit_line[2] + fit_line[0] * width),
              int(fit_line[3] + fit_line[1] * width)), (0, 255, 255), 2)
    # cv2.imshow("img", edge_img)
    # cv2.waitKey(0)
import cv2

import numpy as np

def imgThreshold(img):
    rosource, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(binary, (9, 9), 0)
    return img
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
            elif y2 - y1 == 0:  #水平线保留
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
                                       predict_treshold=0.5, rho=2, theta=np.pi / 180, threshold=50, min_line_length=5,
                                       max_line_gap=2):
        h, w = edge_segmentation.shape
        min_line_length = min_line_length  # minimum number f pixels making up a line
        max_line_gap = 5  # maximum gap in pixels between connectable line segments
        # 检测分割图像的边缘线段
        lines = self.hough_lines(edge_segmentation, rho, theta, threshold, min_line_length, max_line_gap)

        # 去掉垂直、倾角大的线段
        lines = self.delete_line(lines, h, w)
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线


        # 没有检测到吃水线段时，return -1
        if np.sum(lines == None):
            return -1
        line_arr = np.squeeze(lines)
        # 图像中的线段高度由高到低排序
        #line_arr = line_arr[line_arr[:, 1].argsort()][::-1]

        points = self.Collect_points(line_arr)  #线段上的所有点

        #### 保留离图片高度最近的点 ####


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
        flag = True
        # 拟合所有直线段
        if flag:
            # imageOUT = cv2.line(imageOUT, (x0, y0), (x1, y1), (255, 0, 255), 5)
            imageOUT = cv2.line(img,    #img/edge_segmentation
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

        # 过滤垂直的线，必须在二值图上进行操作
        # hline = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h/(h*0.4)), 1))
        # imageOUT = cv2.morphologyEx(imageOUT, cv2.MORPH_OPEN, hline)

        #waterline = cv2.cvtColor(imageOUT, cv2.COLOR_GRAY2RGB) # 转为三通道
        #imgadd = cv2.add(img, waterline)

        save_name = save_dir + img_name + '_fitline3.png'
        cv2.imwrite(save_name, imageOUT)

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

if __name__ == '__main__':

    import os
    img_path = r'D:\Deeplearning\data\ship_test\out\pre'
    img_lst = os.listdir(img_path)
    for img in img_lst:
        img_path_name = os.path.join(img_path, img)
        img_name = img.split('.png')[0]
        img = cv2.imread(img_path_name)

        ## 绘制边缘轮廓 ##
        h, w, _ = img.shape

        ## 绘制轮廓线 ##
        fit = Line_of_horizont_fitting()
        # canny 边缘检测
        pred = cv2.GaussianBlur(img, (3, 3), 0)
        edge = cv2.Canny(pred, 50, 150)  # [h,w]

        # 方案二：检测边缘检测的直线
        lines = fit.hough_lines(edge, 2, np.pi / 180, 50, 15, 30)
        lines = fit.delete_line(lines, h, w)

        # 创建一个吃水线的空白画布
        waterline_img = np.zeros([h, w, 3]).astype(np.uint8)

        # 绘制直线段
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        #     cv2.line(waterline_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
        # # 绘制结果
        # cv2.imshow("img2.jpg", waterline_img)
        # cv2.waitKey(0)


        line_arr = np.squeeze(lines)
        # 图像中的线段高度由高到低排序
        # line_arr = line_arr[line_arr[:, 1].argsort()][::-1]

        points = fit.Collect_points(line_arr)  # 线段上的所有点

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
        flag = True
        # 拟合所有直线段
        if flag:
            # imageOUT = cv2.line(imageOUT, (x0, y0), (x1, y1), (255, 0, 255), 5)
            imageOUT = cv2.line(img,  # img/edge_segmentation
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

        # 过滤垂直的线，必须在二值图上进行操作
        # hline = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h/(h*0.4)), 1))
        # imageOUT = cv2.morphologyEx(imageOUT, cv2.MORPH_OPEN, hline)

        # waterline = cv2.cvtColor(imageOUT, cv2.COLOR_GRAY2RGB) # 转为三通道
        # imgadd = cv2.add(img, waterline)

        save_name = img_name + '_fitline3.png'
        cv2.imwrite(save_name, imageOUT)


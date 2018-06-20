#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import random
import sys
import math


'''
这里主要记录图像特效和线段文字绘制
1. 灰度处理 2. 颜色反转 3. 马赛克 4.毛玻璃 5.图片融合 6.边缘检测 7.浮雕效果 8.颜色增强 9.油画效果
10. 线段绘制 11.形状绘制 12.文字绘制 13.图片绘制
'''

img = cv2.imread(filename='./img/2.jpg', flags=1)

imgInfo = img.shape
# 输出高宽通道数
print(imgInfo)
height = imgInfo[0]
width = imgInfo[1]
# 读取图片如果flags是0的时候，是没有通道的
mode = imgInfo[2]

# 颜色转换，从RGB转换成指定的颜色
# dst = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)

# 通过源码转换颜色(取均值，即（r+g+b）/3)
# dst = np.zeros(imgInfo, np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         (b, g, r) = img[i, j]
#         dst[i, j] = np.uint8((int(b) + int(g) + int(r)) / 3)

# 通过源码转换颜色(通过参数，即（r*0.299+g*0.587+b*0.114)
# dst = np.zeros(imgInfo, np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         (b, g, r) = img[i, j]
#         b = int(b)
#         g = int(g)
#         r = int(r)
#         dst[i, j] = r*0.299+g*0.587+b*0.114

'''
算法优化：1.定点操作比浮点操作要快  2.加减运算比乘除运算要快  3. 移位操作比乘除要快
'''

'''
颜色反转
'''
# 1. 灰度图片的颜色反转(255-图像的灰度值)
# gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# dst = np.zeros(imgInfo, dtype=np.uint8)
# for i in range(0, height):
#      for j in range(0, width):
#         grayPixel = gray[i, j]
#         dst[i, j] = 255 - grayPixel

# 2. 彩色图片的颜色反转(255分别减bgr)
# dst = np.zeros(imgInfo, dtype=np.uint8)
# for i in range(0, height):
#      for j in range(0, width):
#         (b, g, r) = img[i, j]
#         dst[i, j] = (255 - b, 255 - g, 255 - r)

# cv2.imshow(winname='img', mat=dst)

'''
马赛克(用一个像素的颜色替代一定区域的像素颜色)
'''
# for i in range(100, 300):
#     for j in range(100, 200):
#         # 选中一个像素，让这个像素覆盖掉10*10的区域
#         if i % 10 == 0 and j % 10 == 0:
#             for m in range(0, 10):
#                 for n in range(0, 10):
#                     (b, g, r) = img[i, j]
#                     img[i + m, j + n] = (b, g, r)

'''
毛玻璃（在8个像素内随机取一个像素值赋值给当前位置）
'''
# dst = np.zeros(imgInfo, np.uint8)
# mm = 8 # 水平或竖直方向是8
# for i in range(0, height - mm):
#     for j in range(0, width - mm):
#         # 取一个随机数
#         index = int(random.random() * 8) #取一个0~8之间的随机数
#         (b, g, r) = img[i + index, j + index]
#         dst[i, j] = (b, g, r)

'''
图片融合dst = src1*a+src2*(1-a)
  这种方法的实用性有限
'''
# img0 = cv2.imread(filename='./img/image0.jpg', flags=1)
#
# # ROI
# roiH = int(height / 2)
# roiW = int(width / 2)
# img0Roi = img0[0: roiH, 0:roiW]
# imgRoi = img[0: roiH, 0:roiW]
#
# # dst = np.zeros([roiH, roiW, 3], np.uint8)
# dst = cv2.addwidthed(imgRoi, 0.5, img0Roi, 0.5, 0)

'''
边缘检测
  步骤：1. 灰度转化  2. 高斯滤波  3. canny方法
'''
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgG = cv2.GaussianBlur(gray, (3, 3), 0)
# dst = cv2.Canny(imgG, 50, 50)

# 源码实现边缘检测

# sobel 1 算子模版 2 图片卷积 3 阈值判决(超参)

# 1. 算子模板
# [1 2 1          [ 1 0 -1,
#  0 0 0            2 0 -2,
# -1 -2 -1 ]       1 0 -1 ]
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = np.zeros((height, width, 1), np.uint8)
# for i in range(0, height - 2):
#     for j in range(0, width - 2):
#         # 2 图片卷积
#         gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 + gray[i + 2, j] * (-1) + gray[i + 2, j + 1] * (-2) + gray[i + 2, j + 2] * (-1)
#         gx = gray[i, j] * 1 + gray[i + 1, j] * 2 + gray[i + 2, j] * 1 + gray[i, j + 2] * (-1) + gray[i + 1, j + 2] * (-2) + gray[i + 2, j + 2] * (-1)
#         grad = math.sqrt(gy ** 2 + gx ** 2)
#
#         # 阈值判决
#         if grad > 50:
#             dst[i, j] = 255
#         else:
#             dst[i, j] = 0

'''
浮雕效果：
'''
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = np.zeros((height, width, 1), np.uint8)
# for i in range(0, height):
#     for j in range(0, width - 1):
#         grayP0 = int(gray[i, j])
#         grayP1 = int(gray[i, j + 1])
#         # 两个像素的差加一个灰度(150的作用，增强图像的浮雕灰度等级)
#         newP = grayP0 - grayP1 + 150
#         if newP > 255:
#             newP = 255
#         elif newP < 0:
#             newP = 0
#         dst[i, j] = newP

'''
颜色增强
'''
# 原图像
# cv2.imshow(winname='img', mat=img)
# dst = np.zeros((height, width, 3), np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         (b, g, r) = img[i, j]
#         b = int(b) * 1.5
#         g = int(g) * 1.3
#         if b > 255:
#             b = 255
#         if g > 255:
#             g = 255
#         dst[i, j] = (b, g, r)

'''
油画特效
'''
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = np.zeros(imgInfo, np.uint8)
# for i in range(4, height - 4):
#     for j in range(4, width - 4):
#         # 我们想定义的灰度等级是8个，定义一个数组。
#         array1 = np.zeros(8, np.uint8)
#         for m in range(-4, 4):
#             for n in range(-4, 4):
#                 # 把一个像素的0~256之间的值分成8份，看当前像素处于哪个区间
#                 p1 = int(gray[i + m, j + n] / 32)
#                 # 记录每个像素出现在相应区间的次数
#                 array1[p1] = array1[p1] + 1
#
#         currentMax = array1[0]
#
#         l = 0
#         # 遍历array1，取得当前小块区域的像素处于哪个区域最多
#         for k in range(0, 8):
#             if currentMax < array1[k]:
#                 currentMax = array1[k]
#                 l = k
#
#         # 为了简化计算 取均值
#         for m in range(-4, 4):
#             for n in range(-4, 4):
#                 if gray[i + m, j + n] > (l * 32) and gray[i + m, j + n] < ((l + 1) * 32):
#                     (b, g, r) = img[i + m, j + n]
#         dst[i, j] = (b, g, r)

'''
线段绘制
'''
# dst = np.zeros(imgInfo, np.uint8)
# # 参数：img：图片， pt1：线段开始点， pt2：线段终止点， color：线段颜色
# cv2.line(img=dst, pt1=(100, 100), pt2=(300, 300), color=(0, 0, 255))
# # 参数：thickness线条的宽度
# cv2.line(dst, (100, 200), (400, 200), (0, 255, 255), thickness=20)
# # 参数：lineType:线的形状
# cv2.line(dst, (100, 300), (400, 300), (0, 255, 0), 20, lineType=cv2.LINE_AA)
#
# # 绘制一个三角形
# cv2.line(dst, (200, 150), (50, 250), (25, 100, 255))
# cv2.line(dst, (50, 250), (400, 380), (25, 100, 255))
# cv2.line(dst, (400, 380), (200, 150), (25, 100, 255))

'''
其他形状绘制
'''
# dst = np.zeros(imgInfo, np.uint8)
# # 矩形
# # 参数说明：img：图片信息。 pt1：矩形左上角坐标 pt2：矩形右下角坐标。 color：图形的颜色。 thickness：-1. 颜色填充整个图形内部， >=0的时候代表线的宽度
# cv2.rectangle(img=dst, pt1=(50, 100), pt2=(200, 300), color=(255, 0, 0), thickness=1)
#
# # 圆形
# # 参数：img：图片信息 center：圆心坐标 radius：半径 color：图形的颜色。 thickness：-1. 颜色填充整个图形内部， >=0的时候代表线的宽度
# cv2.circle(img=dst, center=(150, 150), radius=(50), color=(0, 255, 0), thickness=3)
#
# # 椭圆
# # 参数：img：图片信息 center：椭圆中心坐标 axes：长短轴长度的一半 angle：椭圆沿逆时针方向旋转的角度
# #       startAngle，endAngle：椭圆弧沿顺时针方向起始的角度和结束角度
# cv2.ellipse(img=dst, center=(300, 300), axes=(200, 100), angle=30, startAngle=0, endAngle=360, color=(255, 255, 0), thickness=2)
#
# # 多边形
# points = np.array([[150, 50], [140, 140], [200, 170], [250, 250], [150, 50]], np.int32)
# print(points.shape)
# points.reshape(-1, 1, 2)
# print(points.shape)
# # 参数：pts：点的集合 isClosed：是否闭合
# cv2.polylines(dst, pts=[points], isClosed=True, color=(0, 255, 255))
# cv2.imshow(winname='dst', mat=dst)

'''
文字图片绘制
'''
# 文字绘制
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.rectangle(img, (100, 200), (400, 500), (0, 255, 0), thickness=1)
# # 参数：1 dst 2 文字内容  org：坐标 fontFace：字体 fontScale：字体大小 color：颜色 thickness：粗细 lineType：样式
# # OpenCV原生函数putText不支持中文字体
# cv2.putText(img, 'OPEN CV', org=(110, 310), fontFace=font, fontScale=1, color=(100, 100, 255), thickness=1, lineType=cv2.LINE_AA)

# 图片绘制
newheight = int(img.shape[0] * 0.2)
newwidth = int(img.shape[1] * 0.2)
imgResize = cv2.resize(img, (newwidth, newheight))
for i in range(0, newheight):
    for j in range(0, newwidth):
        img[i + 200, j + 300] = imgResize[i, j]

cv2.imshow(winname='dst', mat=img)

cv2.waitKey(0)

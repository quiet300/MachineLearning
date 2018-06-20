#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2

import sys

'''
这里是图像的基本操作方法
1. 文件读取  2. 图片缩放  3. 图片剪切  4. 图片移位  5. 图片镜像  6. 图片仿射变换  7. 图片旋转
'''

'''
imread 读取文件，取到的数据是numpy.ndarray类型
参数说明：
  filename：读取的文件名
  flags：0 灰度图片；1 RGB图片
'''
img = cv2.imread(filename='./img/2.jpg', flags=1)

# 图片裁剪
img = img[0:300, 0:300]
'''
图片写入
  filename：生成图片的名称
  img：要生成图片的数据
  params：输出图片质量设置
    1. [cv2.IMWRITE_JPEG_CHROMA_QUALITY, 0] 参数范围0~100 有损压缩。0像素最小
'''
# 输出jpg
# cv2.imwrite(filename='image.jpg', img=img, params=[cv2.IMWRITE_JPEG_CHROMA_QUALITY, 50])

'''
png 的属性，无损。
'''
# 输出png
# cv2.imwrite(filename='image.png', img=img, params=[cv2.IMWRITE_PNG_COMPRESSION, 60])
'''
总结：jpg的时候，0代表压缩比高。jpg有RGB三种颜色构成
      png的时候，0代表压缩比低。png由RGB+alpha构成。alpha是透明度信息
'''
##########################################################

'''
图片中像素的读取写入
'''
#取得宽高100,80位置的像素，返回的是元组类型，格式是bgr
(b, g, r) = img[100, 80]
print(b, g, r)

# 画一道竖线坐标从（10， 100）到（110， 100）
# for i in range(1, 100):
#     img[10 + i, 100] = (0, 0, 255)

# 图片缩放
imgInfo = img.shape
# 输出高宽通道数
print(imgInfo)
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]

'''
等比例缩放和非等比例缩放
  在原来的宽高上乘一个统一的系数就是等比例缩放
  自由设定宽高就是非等比例缩放
'''
# dstHeight = int(height * 0.5)
# dstwidth = int(width * 0.5)

# 默认双线性插值
# dst = cv2.resize(img, (dstwidth, dstHeight))

# 最近邻域插值

# 双线性插值

# cv2.imshow(winname='img', mat=dst)

# 图片剪切(用坐标剪切)
# dst = img[100:200, 100:300]

# cv2.imshow(winname='img', mat=dst)

# 图片移位
'''
位移说明：
  1. 可以先把坐标看成两部分 2*2的矩阵A [1, 0]和[0, 1]和2*1的矩阵B [100, 200]
  2. 原始的图片信息是C x,y
  3. 转换的坐标是A*C + B = [[1*x, 0*y], [0*x, 1*y] + [100,200] = [x+100, y+200] 完成了图片移位
'''
matShift = np.float32([[1, 0, 100], [0, 1, 200]])
'''
  图片移位warpAffine参数说明：
    src：图片信息
    M：移位矩阵
    dsize：图片的info信息
'''
# dst = cv2.warpAffine(img, matShift, (height, width))
# cv2.imshow(winname='img', mat=dst)

# 代码实现（NO API）
# dst = np.zeros(imgInfo, np.int8)
# for i in range(0, height-200):
#     for j in range(0, width-100):
#         dst[i + 200, j+100] = img[i, j]
# cv2.imshow(winname='img', mat=dst)

# 图片镜像
print('图片镜像:',height,width)
newInfo = (height*2, width, mode)
# 关于np.uint8的设置，图片中通常用uint8来设置颜色，不用int8，因为int8会失真
# dst = np.zeros(newInfo, np.uint8)
#
# # 镜像图片
# for i in range(0, height):
#     for j in range(0, width):
#         dst[i, j] = img[i, j]
#         dst[height*2 - i - 1, j] = img[i, j]
#
# # 在图片的中轴上绘制一条直线
# for i in range(0, width):
#     dst[height, i] = (0, 0, 255)
#
# cv2.imshow(winname='img', mat=dst)

'''
图片的仿射变换
  把原图片的坐标点映射到新的坐标上（左上角，左下角，右上角）
'''
# 原图片设置为整个图片
# matsrc = np.float32([[0, 0], [0, height - 1], [width - 1, 0]])
# # 要仿射到的位置
# matdst = np.float32([[50, 50], [100, height - 150], [width - 100, 100]])
#
# # 组合
# matAffine = cv2.getAffineTransform(src=matsrc, dst=matdst)
#
# dst = cv2.warpAffine(img, matAffine, (height, width))
# cv2.imshow(winname='dst', mat=dst)

# 图片旋转
# center:中心点 angle: 旋转的角度 scale:缩放的系数(如果不缩放，旋转后的图片可能超出可观察的范围)
matrorate = cv2.getRotationMatrix2D(center=(height*0.5, width*0.5), angle=45, scale=0.5)
dst = cv2.warpAffine(img, matrorate, (height, width))
cv2.imshow(winname='dst', mat=dst)

'''
imshow 展示图片
参数说明：
  winname：窗口名称
  mat：要显示的图片矩阵
'''
# cv2.imshow(winname='img', mat=img)

'''
waitKey：显示图片时要暂停一下，不然闪一下就没了
0 一直在，0以外 以毫秒计算时间显示
'''
cv2.waitKey(0)




# coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# 打印numpy数组的时候，中间不省略
# np.set_printoptions(threshold=np.inf)

def show_image_tf(image_tf):
    # 需要使用交互式会话
    image = image_tf.eval()
    print(image)
    print('图像大小为:{}'.format(image.shape))

    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], cmap='Greys')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()

# 交互式会话启动
sess = tf.InteractiveSession()

# 读文件夹
dirfload = os.listdir('./image_demo')
for files in dirfload:
    # 文件
    image_path = './image_demo/'+files

    # 1、图像格式的转换
    # 读取数据
    file_contents = tf.read_file(image_path)
    # file_contents是Tensor类型

    # 将图像数据转换为像素点的数据格式，返回对象为: [height（高度）, width（宽度）, num_channels（通道数）],
    #  如果是gif的图像返回[num_frames, height, width, num_channels],num_frames相当于在这个gif图像中有多少个静态图像
    # 参数channels：可选值：0 1 3 4，默认为0， 一般使用0 1 3，不建议使用4
    # 0：使用图像的默认通道，也就是图像是几通道的就使用几通道
    # 1：使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
    # 3：使用RGB三通道读取数据
    # 4：使用RGBA四通道读取数据(R：红色，G：绿色，B：蓝色，A：透明度)
    image_tensor = tf.image.decode_image(file_contents, channels=0)

    im2 = tf.image.decode_jpeg(file_contents, channels=0)

    # image_tensor是Tensor类型
    # 原图
    # show_image_tf(image_tensor)

    # ==================================================================================================================
    # 2、图片大小重置
    # images：要转换的原始image，格式为：[height, width, num_channels]或者[batch, height, width, num_channels]
    # size：要改变为的大小
    # method参数：
    #   BILINEAR 线性插值，默认
    #   NEAREST_NEIGHBOR 最近邻插值，失真最小
    #   BICUBIC 三次插值
    #   AREA 面积插值
    # * 这里的images的值不能是decode_image转换的，只能是具体的decode_png,decode_jpeg等过来的
    # resize_image_tensor = tf.image.resize_images(images=im2, size=(600, 600), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # show_image_tf(resize_image_tensor)

    # 三、图片的剪切&填充
    # 1)、图片重置大小，通过图片的剪切或者填充（从中间开始计算图片的大小）。如果设置的大小比原始图片大，就填充成黑色，否则就剪切图片
    crop_image_tensor = tf.image.resize_image_with_crop_or_pad(image=image_tensor, target_height=800, target_width=800)
    # show_image_tf(crop_image_tensor)

    # 2)、中间等比例剪切（central_fraction是剪切比例）
    center_crop_image_tensor = tf.image.central_crop(image=image_tensor, central_fraction=0.5)
    # show_image_tf(center_crop_image_tensor)

    # 3)、填充给定区域(填充的是黑色)
    pad_to_bounding_box_image_tensor = tf.image.pad_to_bounding_box(image=image_tensor, offset_width=10, offset_height=10, target_width=1000, target_height=1000)
    # show_image_tf(pad_to_bounding_box_image_tensor)

    # 4)、裁剪给定区域
    crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(image=image_tensor, offset_height=0, offset_width=0, target_height=100, target_width=200)
    # show_image_tf(crop_to_bounding_box_image_tensor)

    # 四、旋转
    # 1）、上下交换
    flip_up_down_impage_tensor = tf.image.flip_up_down(image_tensor)
    # show_image_tf(flip_up_down_impage_tensor)

    # 2）、左右交换
    flip_left_right_tensor = tf.image.flip_left_right(image_tensor)
    # show_image_tf(flip_left_right_tensor)

    # 3）、转置(就是矩阵的转置，按对角线)
    transpose_image_tensor = tf.image.transpose_image(image=image_tensor)
    # show_image_tf(transpose_image_tensor)

    # 4）、旋转-逆时针（90度、180度、270度....k*90度）
    rot_image_tensor = tf.image.rot90(image=image_tensor, k=2)
    # show_image_tf(rot_image_tensor)

    # 五、颜色空间的转换（rgb、hsv、gray）
    # 颜色空间的转换必须将image的值转换为float32类型，不能使用unit8类型
    image32_tensor = tf.image.convert_image_dtype(image=image_tensor, dtype=tf.float32)

    # 1）、rgb -> hsv（h: 图像的色彩/色度，s:图像的饱和度，v：图像的亮度）
    rgb_to_hsv_tensor = tf.image.rgb_to_hsv(images=image32_tensor)
    # show_image_tf(rgb_to_hsv_tensor)

    # 2）、hsv -> rgb
    hsv_to_rgb_tensor = tf.image.hsv_to_rgb(images=image32_tensor)
    # show_image_tf(hsv_to_rgb_tensor)

    # 3）、rgb -> gray
    rgb_to_gray_tensor = tf.image.rgb_to_grayscale(images=image32_tensor)
    # show_image_tf(rgb_to_gray_tensor)

    # 4）、可以从颜色空间中提取图像的轮廓信息(图像的二值化)
    # 从灰度图片处理
    a = rgb_to_gray_tensor
    b = tf.less_equal(a, 0.9)
    # 0是黑，1是白
    # condition?true:false
    # condition、x、y格式必须一模一样，当condition中的值为true的之后，返回x对应位置的值，否则返回y对应位置的值
    # 对于a中所有大于0.9的像素值，设置为0
    c = tf.where(condition=b, x=a, y=a-a)
    d = tf.where(condition=b, x=c-c+1, y=c)
    # show_image_tf(d)

    # 六、图像的调整
    # 1）、亮度调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # delta: 取值范围(-1,1）之间的float类型的值，表示对于亮度的减弱或者增强的系数值
    # 底层执行：rgb -> hsv -> h,s,v*delta -> rgb
    brightness_tensor = tf.image.adjust_brightness(image=image_tensor, delta=0.6)
    # show_image_tf(brightness_tensor)

    # 2）、色调调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # delta: 取值范围(-1,1）之间的float类型的值，表示对于色调的减弱或者增强的系数值
    # 底层执行：rgb -> hsv -> h*delta,s,v -> rgb
    adjust_hue_tensor = tf.image.adjust_hue(image=image_tensor, delta=0.3)
    # show_image_tf(adjust_hue_tensor)

    # 3）、饱和度调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # saturation_factor: 一个float类型的值，表示对于饱和度的减弱或者增强的系数值，饱和因子
    # 底层执行：rgb -> hsv -> h,s*saturation_factor,v -> rgb
    adjust_saturation_tensor = tf.image.adjust_saturation(image=image_tensor, saturation_factor=10)
    # show_image_tf(adjust_saturation_tensor)

    # 4）、对比度调整，公式：(x-mean) * contrast_factor + mean
    adjust_contrast_tensor = tf.image.adjust_contrast(images=image_tensor, contrast_factor=10)
    # show_image_tf(adjust_contrast_tensor)

    # 5）、图像的gamma校正
    # images: 要求必须是float类型的数据**
    # gamma：任意值，Oup = In * Gamma
    adjust_gamma_tensor = tf.image.adjust_gamma(image=image32_tensor, gamma=30)
    # show_image_tf(adjust_gamma_tensor)

    # 6）、图像的归一化(x-mean)/adjusted_sttdev, adjusted_sttdev=max(stddev, 1.0/sqrt(image.NumElements()))
    #     *per_image_standardization返回的类型是float类型的，但是图片展示需要的是整形的，所以用cast做一个转换
    per_image_tensor = tf.cast(tf.image.per_image_standardization(image_tensor), dtype=tf.uint8)
    show_image_tf(per_image_tensor)
    # ValueError: Floating point image RGB values must be in the 0..1 range.

    # 七、噪音数据的加入
    noise_image_tensor = image_tensor + tf.cast(7*tf.random_normal(shape=[362, 500, 3], mean=0.0, stddev=0.1), dtype=tf.uint8)
    # show_image_tf((noise_image_tensor))

    break





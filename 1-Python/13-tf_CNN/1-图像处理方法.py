# -- encoding:utf-8 --


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# -》np.set_printoptions(threshold=np.inf)：打印大量的numpy数组对象时，保证数据显示完整（中间不会有...） ★
np.set_printoptions(threshold=np.inf)

'''
    1、定义图像展示方法
    -》imshow(*)：显示图像
'''
def show_image_tensor(image_tensor):   # 传入tensor图像对象
    # print(image_tensor)   # tensor图像说明：Tensor("add_3:0", shape=(600, 510, 3), dtype=uint8) unit8为0-255的RGB数值

    image = image_tensor.eval()   # -》.eval()：将string/tensor转为list，将tensor图像转为list数据[h,w,c]  ★
    print("图像大小为:{}".format(image.shape))   # 图像大小为:(600, 510, 3)

    if len(image.shape) == 3 and image.shape[2] == 1:   # ①、黑白图像，c=1
        plt.imshow(image[:, :, 0], cmap='Greys_r')   # Greys_r黑白颜色。不加Greys_r时会显示为热力图
        plt.show()
    elif len(image.shape) == 3:   # ②、彩色图像，三个维度均有值
        plt.imshow(image)
        plt.show()

'''
    2、启动交互式会话
    -》tf.InteractiveSession()
'''
sess = tf.InteractiveSession()

'''
    3、读取图像
    -》tf.read_file(path)
'''
image_path = 'data/xiaoren.png'
# image_path = 'data/gray.png'
# image_path = 'data/black_white.jpg'

file_contents = tf.read_file(image_path)

'''
    4、显示图像   ★
    -》tf.image.decode_image(contents,channels)：图像解码，将图像数据转换为像素点的数据格式，这样才可以对图像进行操作
        其有多种形式，输入tf.image.即可查看对应的专用图像解码器
    
    ①、输入图像为BMP, JPEG, PNG时，返回值为 [height, width, num_channels]
    ②、输入图像为gif时，返回值为 [num_frames, height, width, num_channels]
        height: 高度像素
        width: 水平宽度像素
        num_channels: 图像的通道数，即channels的值
        num_frames: 因为gif的图像是一个动态图像，可以将每一个动的画面看成一个静态图像，num_frames相当于在这个gif图像中有多少个静态图像
    
    channels：可选值：0 1 3 4，默认为0， 一般使用0 1 3
    0：使用图像的默认通道，也就是图像是几通道的就使用几通道
    1：使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
    3：使用RGB三通道读取数据
    4：使用RGBA四通道读取数据(R：红色，G：绿色，B：蓝色，A：透明度)
'''
image_tensor = tf.image.decode_png(contents=file_contents, channels=3)   # 方法一，对png专用的图像解码
# image_tensor = tf.image.decode_image(contents=file_contents, channels=3)   # 方法二：对大部分图像的解码，但有时候无效
# show_image_tensor(image_tensor)


"""
    （一）图像大小重置
    -》tf.image.resize_images(image,size,method)：返回值和images格式一样，唯一区别是height和width变化为给定的值
        images: tensor对象，格式为[height, width, num_channels]或者[batch, height, width, num_channels]
        batch：tensor为多图时，批量地更新图片大小

        method：
        -》tf.image.ResizeMethod.
            BILINEAR = 0 ：线性插值，默认
            NEAREST_NEIGHBOR = 1 ：最近邻插值，失真最小，常用
            BICUBIC = 2 ：三次插值
            AREA = 3 ：面积插值
"""
resize_image_tensor = tf.image.resize_images(images=image_tensor, size=(200, 200),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# show_image_tensor(resize_image_tensor)

'''
    （二）图像的剪切或填充
'''
# ①、-》tf.image.resize_image_with_crop_or_pad()：从中间开始剪切
crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor, 200, 200)
# show_image_tensor(crop_or_pad_image_tensor)

# ②、-》ttf.image.central_crop()：从中间开始等比例（按图像比例）剪切
central_crop_image_tensor = tf.image.central_crop(image_tensor, central_fraction=0.2)
# show_image_tensor(central_crop_image_tensor)

# ③、-》tf.image.crop_to_bounding_box()：在给定位置剪切
# offset_height和offset_width：图像相对于左上角点的偏移
crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(image_tensor, offset_height=10, offset_width=40,target_height=200, target_width=300)
# show_image_tensor(crop_to_bounding_box_image_tensor)

# ④、-》tf.image.pad_to_bounding_box()：在给定位置填充，在图像周边填充黑色像素
pad_to_bounding_box_image_tensor = tf.image.pad_to_bounding_box(image_tensor, offset_height=400, offset_width=490,target_height=1000,target_width=1000)
# show_image_tensor(pad_to_bounding_box_image_tensor)


'''
    （三）旋转
'''
# ①、-》tf.image.flip_up_down()：上下交换
flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# show_image_tensor(flip_up_down_image_tensor)

# ②、-》tf.image.flip_left_right()：左右交换
flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# show_image_tensor(flip_left_right_image_tensor)

# ③、-》tf.image.transpose_image()：转置，注意不是旋转
transpose_image_tensor = tf.image.transpose_image(image_tensor)
# show_image_tensor(transpose_image_tensor)

# ④、-》tf.image.rot90()：旋转（90度、180度、270度....）
k_rot90_image_tensor = tf.image.rot90(image_tensor, k=1)   # 旋转角度为 = k*90度旋转，逆时针旋转
# show_image_tensor(k_rot90_image_tensor)

'''
    （四）颜色空间的转换（rgb、hsv、gray）
'''
# ①、-》tf.image.convert_image_dtype()：颜色空间的转换必须将image的值转换为float32类型（相当于除255），不能使用unit8类型 - 原tensor图为unit8类型
float32_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# show_image_tensor(float32_image_tensor)

# ②、-》tf.image.rgb_to_hsv()：rgb -> hsv（h: 图像的色彩/色度，s:图像的饱和度，v：图像的亮度）
hsv_image_tensor = tf.image.rgb_to_hsv(float32_image_tensor)
# show_image_tensor(hsv_image_tensor)

# ③、-》tf.image.hsv_to_rgb()：hsv -> rgb   转彩色
rgb_image_tensor = tf.image.hsv_to_rgb(hsv_image_tensor)
# show_image_tensor(rgb_image_tensor)

# ④、-》tf.image.rgb_to_grayscale()：rgb -> gray   转黑白
gray_image_tensor = tf.image.rgb_to_grayscale(rgb_image_tensor)
# show_image_tensor(gray_image_tensor)

# ⑤、从颜色空间中提取图像的轮廓信息 - 图像的二值化。图像的二值化配合图像剪切，可以获得剪切图像
a = gray_image_tensor   # 轮廓提取通常用灰白图
b = tf.less_equal(a, 0.9)   # 返回bool值。切分值越小，轮廓越模糊

# 分割轮廓，a中小于等于0.9的像素值，设置为1 - 对应白；大于0.9的像素点，设置为0 - 对应黑
# condition，x，y的格式必须一致，所以这里不能直接写 x=1,y=0，而是做一个转换，使x和y为一个tensor图像格式
c = tf.where(condition=b, x= a-a+1, y=a-a)
# show_image_tensor(c)

#多次二值化可以得到更局部的轮廓
# b2 = tf.less_equal(a, 0.7)
# d = tf.where(condition=b2,x=c-c+1,y=c-c)

'''
    （五）图像的调整
'''
# ①、-》tf.image.adjust_brightness()：亮度调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # delta: 取值范围(-1,1）之间的float类型的值，表示对于亮度的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h,s,v*delta -> rgb
adjust_brightness_image_tensor = tf.image.adjust_brightness(image=image_tensor, delta=0.8)
# show_image_tensor(adjust_brightness_image_tensor)

# ②、-》tf.image.adjust_hue()：色调调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # delta: 取值范围(-1,1）之间的float类型的值，表示对于色调的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h*delta,s,v -> rgb
adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=-0.8)
# show_image_tensor(adjust_hue_image_tensor)

# ③、-》tf.image.adjust_saturation()：饱和度调整
    # image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
    # saturation_factor: 一个float类型的值，表示对于饱和度的减弱或者增强的系数值，饱和因子
# 底层执行：rgb -> hsv -> h,s*saturation_factor,v -> rgb
adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=20)
# show_image_tensor(adjust_saturation_image_tensor)

# ④、-》tf.image.adjust_contrast()：对比度调整，公式：(x-mean) * contrast_factor + mean  偏移量+均值 理解为小的更小，打得更大
adjust_contrast_image_tensor = tf.image.adjust_contrast(image_tensor, contrast_factor=10)
# show_image_tensor(adjust_contrast_image_tensor)

# ⑤、-》tf.image.adjust_gamma()：图像的gamma校正
# images: 要求必须是float类型的数据
# gamma：任意值，Out = In * Gamma 理解为只要不是白色，就加深颜色
adjust_gamma_image_tensor = tf.image.adjust_gamma(float32_image_tensor, gamma=100)
# show_image_tensor(adjust_gamma_image_tensor)

# ⑥、-》tf.image.per_image_standardization()：图像的归一化，放置梯度消失(x-mean)/adjusted_sttdev, adjusted_sttdev=max(stddev, 1.0/sqrt(image.NumElements()))  避免标准差为0
per_image_standardization_image_tensor = tf.image.per_image_standardization(image_tensor)
# show_image_tensor(per_image_standardization_image_tensor)

'''
    （六）噪音数据的加入
'''
noisy_image_tensor = image_tensor + tf.cast(5 * tf.random_normal(shape=[600, 510, 3], mean=0, stddev=0.1), tf.uint8)   # 将随机数转为unit8类型
# show_image_tensor(noisy_image_tensor)

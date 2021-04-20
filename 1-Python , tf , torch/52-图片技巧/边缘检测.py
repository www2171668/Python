"""
cv2.Canny(image,            # 输入原图（必须为单通道图）
          threshold1,
          threshold2,       # 较大的阈值2用于检测图像中明显的边缘
          [, edges[,
          apertureSize[,    # apertureSize：Sobel算子的大小
          L2gradient ]]])   # 参数(布尔值)：
                              true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                              false：使用L1范数（直接将两个方向导数的绝对值相加）。
"""

import cv2

# 读取灰度图
original_img = cv2.imread("tuxiang2.png", 0)                              # 读取图像

# canny(): 边缘检测
img1 = cv2.GaussianBlur(original_img, (3, 3), 0)                            # 高斯模糊级别。(3, 3)表示高斯矩阵的长宽值，标准差取0
canny = cv2.Canny(img1, 50, 150)

# 形态学：边缘检测。并不是总是有效
_, Thr_img = cv2.threshold(original_img, 210, 255, cv2.THRESH_BINARY)       # 设定红色通道阈值210（阈值影响梯度运算效果）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                  # 定义矩形结构元素
gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)            # 梯度

cv2.imshow("original_img", original_img)                                   # 前面的名字不能重复
# cv2.imshow('Canny', canny)
# cv2.imshow("gradient", gradient)

original_img = cv2.imread("tuxiang2.png")
img_man = original_img[30:,:,:]
cv2.imshow("imgman", img_man)

# contours, hierarchy = cv2.findContours(original_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours,hierarchy)
# print('_'*40)

contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       # 检测轮廓
cv2.drawContours(original_img,contours,-1,(0,0,255),3)      # 绘制轮廓
cv2.imshow("img_new", original_img)

# print('_'*40)
# contours, hierarchy = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours,hierarchy)
# print('_'*40)

cv2.waitKey(0)
cv2.destroyAllWindows()
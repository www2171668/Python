import cv2
import numpy as np

img = cv2.imread("tuxiang2.png")
cv2.imshow("image1",img)

gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 颜色空间转换函数，转为灰度图
h, w = 110, 84
gray_resized = cv2.resize(gray_image, (w, h))       # 缩放
cv2.imshow("image2",gray_resized)

gray_resized = gray_resized[13:h - 13, :]  # 先修剪图片，再取其中一部分并修剪图片没用的上下边缘：110-13-13 = 84。
cv2.imshow("image3",gray_resized)

gray_resized2 = cv2.resize(gray_image, (84, 84))       # 缩放
cv2.imshow("image4",gray_resized2)

# gray_reshaped = gray_resized.reshape((1, 84, 84))  # 增加batch维度
# cv2.imshow("image3",gray_reshaped)

cv2.waitKey(0)
cv2.destroyAllWindows()
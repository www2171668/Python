import cv2
import numpy as np

#创建 长度（高度）：100, 宽度：600 全部为0 的三维数组
img = np.zeros((100,600,3), dtype = np.uint8)
#显示图片 600×600 全黑的图片
cv2.imshow("img_before", img)

#x方向 0到200像素列  B 蓝色通道 全部置为 255
img[:, 0:200, 0] = 0
#x方向 200到400像素列  G 绿色通道 全部置为 255
img[:, 200:400, 1] = 255
#x方向 400到600像素列  R 绿色通道 全部置为 255
img[:, 400:600, 2] = 255

#打印图片 数组数据
print("Image = \n", img)

#显示图片 三色图片
cv2.imshow("img_after", img)

#等待按键输入 关闭窗口
cv2.waitKey (0)
cv2.destroyAllWindows()
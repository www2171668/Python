import cv2
import numpy as np

img = cv2.imread("tuxiang1.png", 0)
cnt = cv2.imread("tuxiang2.png", 0)
# 用绿色(0, 255, 0)来画出最小的矩形框架
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 用红色表示有旋转角度的矩形框架
# rect = cv2.minAreaRect(cnt)
# box = cv2.cv.BoxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
# cv2.imwrite('contours.png', img)


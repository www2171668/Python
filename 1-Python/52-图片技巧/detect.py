import cv2
import numpy as np
from PIL import Image
from pylab import  *

'''
     key
'''

img = cv2.imread("tuxiang2.png")                            # 返回的img是numpy
template = cv2.imread("ladder.png")                             # (20, 10, 3) -> 转灰度图为(20, 10)
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                # 灰度图
res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)     # 进行图像匹配度，找出匹配区域
# TM_CCOEFF_NORMED为相关系数匹配法：1表示完美的匹配；-1表示最差的匹配
# 必须要正img和template的色度是一致的

w,h = template.shape[1],template.shape[0]                                  # 逆转

threshold = 0.8
loc = np.where(res >= threshold)

###########         在区域内画图         ###########
# for pt in zip(loc[1], loc[0]):
#      cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imshow("image",img)

# 截断子目标图
subgoal_set = []
for pt in zip(loc[1], loc[0]):
     p = (pt[0], pt[1])
     cv2.rectangle(img, p, (p[0] + w, p[1] + h // 2), (0, 0, 255), 2)
     p = ( pt[0], pt[1]+h//2 )
     cv2.rectangle(img, p, (p[0] + w, p[1] + h // 2), (0, 0, 255), 2)
     p = (pt[0], pt[1] - h // 2)
     cv2.rectangle(img, p, (p[0] + w, p[1] + h // 2), (0, 0, 255), 2)

cv2.imshow("image",img)

###########         在图中画点          ###########
# im = array(Image.open('tuxiang2.png'))
# imshow(im)
# plot(loc[1], loc[0], 'r*')              # 得到钥匙的左上角。使用红色星状物标记绘制点 —— 在图像上画点
# plot(loc[1], loc[0] + h//2, 'r*')              # 得到钥匙的左上角。使用红色星状物标记绘制点 —— 在图像上画点
# show()

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
     door
'''
# img = cv2.imread("tuxiang2.png")
# template = cv2.imread('door.png')  # 读取obj图像
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#
# threshold = 0.95
# loc_np = np.where(res >= threshold)  # 根据图像匹配度，得到obj像素索引集合
# loc = (loc_np[1], loc_np[0])  # 转化obj像素索引为图像坐标集合
#
# im = array(Image.open('tuxiang2.png'))
# imshow(im)
#
# plot(loc[0][0], loc[1][0], 'r*')        # 得到door的左上角。因为门的图包括了上面的横梁，所以会大一点
# plot(loc[0][1], loc[1][1], 'r*')
# show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
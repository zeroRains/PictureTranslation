import os
import cv2

path = 'gradually/'
filelist = os.listdir(path)
fps = 1  # 视频每秒24帧
size = (470, 601)  # 需要转为视频的图片的尺寸
# 可以使用cv2.resize()进行修改
video = cv2.VideoWriter("gradual1.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
# 视频保存在当前目录下
for item in filelist:
    if item.endswith('.jpg'):
        # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
        item = path + item
        img = cv2.imread(item)
        video.write(img)
video.release()
cv2.destroyAllWindows()

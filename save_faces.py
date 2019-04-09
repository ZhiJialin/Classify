

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os

import sys

import tensorflow as tf




# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

# 读取图像的路径
path_read = "data/picture/"





# 用来存储生成的单张人脸的路径
path_save = "data/other_faces/"

# Delete old images
def clear_images():
    imgs = os.listdir(path_save)

    for img in imgs:
        os.remove(path_save + img)

    print("clean finish", '\n')


clear_images()

size = 64
index=1
for (path, dirnames, filenames) in os.walk(path_read):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # 从文件读取图片
            img = cv2.imread(img_path)
            faces = detector(img, 1)
            print("人脸数：", len(faces), '\n')
            for k, d in enumerate(faces):

                # 计算矩形大小
                # (x,y), (宽度width, 高度height)
                pos_start = tuple([d.left(), d.top()])
                pos_end = tuple([d.right(), d.bottom()])

                # 计算矩形框大小
                height = d.bottom() - d.top()
                width = d.right() - d.left()

                # 根据人脸大小生成空的图像
                img_blank = np.zeros((height, width, 3), np.uint8)

                for i in range(height):
                    for j in range(width):
                        img_blank[i][j] = img[d.top() + i][d.left() + j]

                # cv2.imshow("face_"+str(k+1), img_blank)

                # 存在本地
                print("Save to:img_face_" + str(k + 1) + ".jpg")
                cv2.imwrite(path_save + str(k + 1) + ".jpg", img_blank)

            # # 转为灰度图片
            # # frame_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # frame_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # 使用detector进行人脸检测 dets为返回的结果
            # dets = detector(frame_new, 1)
            #
            # #使用enumerate 函数遍历序列中的元素以及它们的下标
            # #下标i即为人脸序号
            # #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            # #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            # for i, d in enumerate(dets):
            #     x1 = d.top() if d.top() > 0 else 0
            #     y1 = d.bottom() if d.bottom() > 0 else 0
            #     x2 = d.left() if d.left() > 0 else 0
            #     y2 = d.right() if d.right() > 0 else 0
            #     # img[y:y+h,x:x+w]
            #
            #     face = img[x1:y1,x2:y2]
            #     # 调整图片的尺寸
            #     face = cv2.resize(face, (size,size))
            #     # cv2.imshow('image',face)
            #     # 保存图片
            #     cv2.imwrite(path_save+'/'+str(index)+'.jpg', face)
            #     index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)


#
# img = cv2.imread(path_read+"150.jpg")
# # Dlib 检测
# faces = detector(img, 1)
#
# print("人脸数：", len(faces), '\n')
#
# for k, d in enumerate(faces):
#
#     # 计算矩形大小
#     # (x,y), (宽度width, 高度height)
#     pos_start = tuple([d.left(), d.top()])
#     pos_end = tuple([d.right(), d.bottom()])
#
#     # 计算矩形框大小
#     height = d.bottom()-d.top()
#     width = d.right()-d.left()
#
#     # 根据人脸大小生成空的图像
#     img_blank = np.zeros((height, width, 3), np.uint8)
#
#     for i in range(height):
#         for j in range(width):
#                 img_blank[i][j] = img[d.top()+i][d.left()+j]
#
#     # cv2.imshow("face_"+str(k+1), img_blank)
#
#     # 存在本地
#     print("Save to:img_face_"+str(k+1)+".jpg")
#     cv2.imwrite(path_save+"img_face_"+str(k+1)+".jpg", img_blank)

import os
import cv2
import face_recognition
import matplotlib as PIM
import shutil
import random
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import tensorflow as tf
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 第一步：从存有所有图片的数据集中取出第一张图片
def get_one_picture(rootfile):
    #读取文件下的所有图片
    for filename in os.listdir(rootfile):
        return filename

# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

# #移动图片
# def p2p(picture1_path,new_picture2_path):
#
#     pathDir = os.listdir(picture1_path)  # 取图片的原始路径
#     # filenumber = len(pathDir)
#     # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#     # picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
#     # sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
#     # print(sample)
#     # for name in sample:
#     #     shutil.move(picture1 + name, picture2 + name)
#     return



# 第二步：把移动的图片作为一个训练集进行训练
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) == 0:
                return False

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

#使用CNN提取特征进行训练
def train_CNN():


    return True


# 预测图片
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.62):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



# 判断文件是否为空
def isEmpty(file):
    files = os.listdir(file)
    # 文件下存在内容，返回true，表示不为空
    if files:
        return True
    # 文件下不存在内容，返回false，表示
    # 为空
    else:
        return False



count = 110
path  = r"data/other_faces"
path1 = r"data/分类结果"
while(isEmpty(path)):
    count_times = 0
    # 第一步：从数据集的文件中取出第一张图片
    #创建临时文件夹

    Createfile_temp = r"G:\TensorFlow\人脸识别\data\temp"+"\\"+str(count)
    mkdir(Createfile_temp)
    #获取数据集的第一张图片
    rootfile = r"G:\TensorFlow\人脸识别\data\other_faces"
    image = get_one_picture(rootfile)
    # #将第一张图片存到临时文件夹中
    shutil.move(rootfile+"\\"+image,Createfile_temp+"\\"+image)

    # image = cv2.imread(get_one_picture(r"G:\TensorFlow\人脸识别\data\picture"))


    #训练数据集
    classifier = train("data/temp", model_save_path="trained_knn_temp_model.clf", n_neighbors=1)
    if classifier==False:
        shutil.rmtree(Createfile_temp)
        continue



    #预测图片
    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("data/other_faces"):
        full_file_path = os.path.join("data/other_faces", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_temp_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            if name == str(count):
                shutil.move(rootfile+"\\"+image_file,Createfile_temp+"\\"+image_file)
                count_times+=1
                # if count_times<=2:
                #     # 训练数据集
                #     classifier = train("data/temp", model_save_path="trained_knn_temp_model.clf", n_neighbors=1)




        # # Display results overlaid on an image
        # show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
        # if name == 'lkq':
        #     lkq_count += 1
        # elif name == 'woman':
        #     woman_count += 1
    # print("李克强在视频中出现了%d次" % lkq_count)
    # print("woman在视频中出现了%d次" % woman_count)
    print("第%d类共出现了%d次"%(count,count_times))
    shutil.move(Createfile_temp, path1)
    time.sleep(1)
    count+=1


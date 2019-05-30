import cv2
from PIL import Image
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3    # 对传入神经网络的图像进行预处理
IMAGE_H, IMAGE_W = 416, 416


def crop(image):
    return image[90:-50, :, :]   # get roi


def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)   # resize to fit the NN


def rgb2yuv(image):  # 将RGB图像转化为YUV通道的图像
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def process(frame):
    resized = cv2.resize(frame, (320, 240))
    radar = cv2.cvtColor(resized[212:232, 22:42, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]

    roi = resized[90:-50, :, :]
    roi = cv2.resize(roi, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)

    return roi_yuv, radar


def yolo_img_process(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
    img_resized = img_resized / 255.
    return img_resized
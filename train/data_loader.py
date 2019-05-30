import gzip
import pickle
from deepgtav.messages import frame2numpy
import cv2
import numpy as np
import math

dataset_path = "dataset/dataset.pz"   # 设置数据集位置


def load_data(verbose=1, samples_per_batch=1000):
    batch_count = 0
    dataset = gzip.open(dataset_path)

    while True:
        try:

            x_image_train = []
            x_speed_train = []
            x_image_test = []
            x_speed_test = []

            y_train = []
            y_test = []

            count = 0
            print('----------- On Batch: ' + str(batch_count) + ' -----------')
            while count < samples_per_batch:
                data_dct = pickle.load(dataset)
                frame = data_dct['frame']
                image = frame2numpy(frame, (320, 240))
                image = cv2.GaussianBlur(image, (5, 5), 0)
                image = image[130:240, 0:320]
                image = cv2.GaussianBlur(image, (5, 5), 0)
                speed = data_dct['speed']

                steering = data_dct['steering']
                steering = steering*5/math.pi
                brake = data_dct['brake']
                throttle = data_dct['throttle']

                # Train test split
                if (count % 8) != 0:  # Train   #抽取80%为训练集 20%作为测试集。。。
                    x_image_train.append(image)
                    x_speed_train.append(np.array(float(speed)))

                    y_train.append(np.array([float(steering), float(throttle), float(brake)]))

                else:  # Test
                    x_image_test.append(image)
                    x_speed_test.append(np.array(float(speed)))

                    y_test.append(np.array([float(steering), float(throttle), float(brake)]))

                count += 1
                if (count % 250) == 0 and verbose == 1:
                    print('     ' + str(count) + ' data points loaded in batch.')

            x_train = [np.array(x_image_train), np.array(x_speed_train)]
            x_test = [np.array(x_image_test), np.array(x_speed_test)]
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            print('Batch loaded.')
            batch_count += 1
            yield x_train, y_train, x_test, y_test   # 将y打包为numpy数组  将x打包为list传递至  train

        except EOFError:
            print("end")
            break


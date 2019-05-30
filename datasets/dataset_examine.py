import pickle
import gzip
import cv2
from deepgtav.messages import frame2numpy
import os
from tqdm import tqdm

data_num = 1  # 检查的数据集的编号.....程序以二进制文件的形式将筛选出的数据组组成list存储下来

file = gzip.open('D:/no_traffic/dataset'+str(data_num)+'.pz', 'rb')
deleted_list_pos = "datacleaned/dataset"+str(data_num)+".pickle"
file_exist = os.path.exists(deleted_list_pos)

delete_data = []
last_ele = 0

pointer = 0

if file_exist:
    deleted_list = open(deleted_list_pos, 'rb')
    delete_data = pickle.load(deleted_list)
    last_ele = delete_data[-1]
    pointer = last_ele-1
    # print(delete_data)
    deleted_list.close()
    for i in tqdm(range(last_ele-1)):  # list删除项是100，迭代99，即可看到删除图像
        pickle.load(file)

while True:
    try:
        data_dict = pickle.load(file)
        speed = data_dict['speed']
        throttle = data_dict['throttle']
        brake = data_dict['brake']
        location = data_dict["location"]
        print(str(speed)+", "+str(throttle)+","+str(brake))
        frame = data_dict['frame']
        # Show full image
        image = frame2numpy(frame, (320, 240))
        pointer = pointer+1
        while True:
            cv2.imshow('img', image)
            if cv2.waitKey(0) & 0xFF == 32:
                break
            elif cv2.waitKey(0) & 0xFF == 8:
                delete_data.append(pointer)
                print("##### delete "+str(pointer)+" successfully ####")
                break
            elif cv2.waitKey(0) & 0xFF == ord('q'):
                delete_data.append(pointer)
                print("====================done===================")
                pickle.dump(delete_data, open(deleted_list_pos, "wb"))
                file.close()
                exit(0)

    except EOFError:
        print("===========data collection end=============")
        delete_data.append(pointer+1)
        pickle.dump(delete_data, open(deleted_list_pos, "wb"))
        file.close()
        exit(0)

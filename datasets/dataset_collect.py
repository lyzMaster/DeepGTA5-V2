import time
import requests
import multiprocessing
import numpy as np
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client
import pickle


i = 100   # 设置数据集编号
set_time = 24  # 设置模拟器中的时间。24小时制
weather = 'EXTRASUNNY'  # 设置模拟器中的天气。可以为雨天.....
location = [1243.9920654296875, -1186.1495361328125, 48.48838806152344]  # 设置在模拟器中车辆的初始位置

left = total = right = 0

url = "https://www.pixeldesert.com/compare"  # 设置远程遥控器的地址。
txt_position = "../state.config"
tongji_position = "tongji/"+str(i)+".txt"  # 设置统计文件的位置。统计文件的作用是

CAP_IMG_W, CAP_IMG_H = 320, 240  # 设置要采集的数据集中图像的尺寸

rate = 10  # 设置图像采集的频率，即每秒钟采集多少组数据
client = None
scenario = None

dataset = Dataset(rate=10,
				  frame=[CAP_IMG_W,CAP_IMG_H],
				  throttle=True,
				  brake=True,
				  steering=True,
				  location=True,
				  drivingMode=True,
				  speed=True,
				  time=True)

def state():
    while True:
        response = requests.request("POST", url)
        fo = open(txt_position, "w")
        if response.text[1] == '0':
            fo.write("0")
        elif response.text[1] == '1':
            fo.write("1")
        fo.close()
        time.sleep(2)


def drive():
    global message
    global left, right, total
    message = client.recvMessage()
    speed = message['speed']
    steering = message['steering']
    brake = message['brake']
    location = message["location"]

    total = total + 1
    if steering > 0.1:
        right = right+1
    elif steering < -0.1:
        left = left+1

    print(str(speed)+", "+str(steering)+","+str(brake))
    print(location)


def main():
    global client, scenario

    client = Client(ip='localhost', port=8000,datasetPath = 'D:/no_traffic/dataset' + str(i) + '.pz',compressionLevel=9)
    scenario = Scenario(weather=weather, vehicle='blista', time=[set_time, 0], drivingMode=-1,
                        location=location)   # 设置数据集存储位置，采集数据所用的车辆型号
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    print("load deepGTAV successfully! \nbegin")

    while True:
        fo = open(txt_position, "r")  # 配置1
        txt = fo.read()
        fo.close()
        if txt == '0':
            tongji = open(tongji_position, "w+")
            tongji.write("left:"+str(left)+" right:"+str(right)+" total:"+str(total))
            tongji.close()
            print('=====================end=====================')
            exit(0)
        elif txt == '1':
            drive()


if __name__ == '__main__':
    while True:
        response = requests.request("POST", url)
        if response.text[1] == '1':
            fo = open(txt_position, "w")
            fo.write("1")
            fo.close()
            print('$$$$$$$$$$$$$$$$$$$ after 3s will begin $$$$$$$$$$$$$$$$$$$')
            i = 3
            while i > 0:
                print(i)
                i = i - 1
                time.sleep(1)
            print('=====================start now!======================')
            break
        print("waiting for instructions...")
        time.sleep(1)
    p1 = multiprocessing.Process(target=state)
    p2 = multiprocessing.Process(target=main)
    p1.start()
    p2.start()

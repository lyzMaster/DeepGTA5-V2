import time
import requests
import multiprocessing
import numpy as np
from keras.models import load_model
from utils.img_process import process, yolo_img_process
import utils.yolo_util as yolo_util
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client
import tensorflow as tf
import cv2
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

gamepad = None

url = "https://www.pixeldesert.com/compare"
config_position = "state.config"
client = None
scenario = None
IMAGE_H, IMAGE_W = 416, 416
CAP_IMG_W, CAP_IMG_H = 320, 240

dataset = Dataset(rate=10,
                  frame=[CAP_IMG_W, CAP_IMG_H],
                  throttle=True,
                  brake=True,
                  steering=True,
                  location=True,
                  drivingMode=True,
                  speed=True,
                  time=True
                  )


def state():
    while True:
        response = requests.request("POST", url)
        fo = open(config_position, "w")
        if response.text[1] == '0':
            fo.write("0")
        elif response.text[1] == '1':
            fo.write("1")
        fo.close()
        time.sleep(2)


def set_gamepad(control, throttle, breakk):
    if control == -1:         # stop the car
        client.sendMessage(Commands(0.0, 1, control))
        return

    client.sendMessage(Commands(float(throttle), float(breakk), float(control)))
    # print(str(control)+","+str(throttle)+","+str(breakk))
    # client.sendMessage(Commands(1, 0, 0))


def drive(model, image, speed, warning):

    throttle = 0
    breakk = 0

    roi, radar = process(image)

    controls = model.predict([np.array([roi]), np.array([radar]), np.array([speed])], batch_size=1)
    controls = controls[0][0]*5/3.14

    if controls > 0:
        controls = controls
        print("-->" + "   speed=" + str(speed)+",controls="+str(controls))
    else:
        print("<--" + "   speed=" + str(speed)+",controls="+str(controls))

    if warning:
        return controls, 0, 1

    if speed < 5:       # control speed
        throttle = 1
    elif speed < 20:
        throttle = 0.5
    elif speed > 25:
        throttle = 0.0
        breakk = 0.4

    return controls, throttle, breakk


def main():

    global client, scenario
    client = Client(ip='localhost', port=8000)  # Default interface
    scenario = Scenario(weather='EXTRASUNNY', vehicle='blista', time=[12, 0], drivingMode=-1
                        , location=[-2583.6708984375, 3501.88232421875, 12.7711820602417])
    client.sendMessage(Start(scenario=scenario, dataset=dataset))
    print("load deepGTAV successfully! \nbegin")

    # load yolo v3
    classes = yolo_util.read_coco_names('./files/coco/coco.names')
    num_classes = len(classes)
    input_tensor, output_tensors = yolo_util.read_pb_return_tensors(tf.get_default_graph(),
                                                                    "./files/trained_models/yolov3.pb",
                                                                    ["Placeholder:0", "concat_9:0", "mul_6:0"])
    print("load yolo v3 successfully!")

    with tf.Session() as sess:
        model = load_model("files/trained_models/main_model.h5")
        print("load main_model successfully!")
        while True:
            fo = open(config_position, "r")  # 配置1
            txt = fo.read()
            fo.close()
            if txt == '0':
                set_gamepad(-1, -1, 0)
                time.sleep(0.7)
                print('=====================end=====================')
                exit(0)
            elif txt == '1':
                message = client.recvMessage()
                frame = frame2numpy(message['frame'], (CAP_IMG_W, CAP_IMG_H))
                image_obj = Image.fromarray(frame)



                speed = message['speed']

                boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(yolo_img_process(frame), axis=0)})
                boxes, scores, labels = yolo_util.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.1)
                image, warning = yolo_util.draw_boxes(image_obj, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

                control, throttle, breakk = drive(model=model, image=frame, speed=speed, warning=warning)

                print(warning)

                set_gamepad(control, throttle, breakk)
                # show
                # result = np.asarray(image)
                # info = "warning:%s" % (str(warning))
                # cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1, color=(255, 0, 0), thickness=2)
                # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                # cv2.imshow("result", result)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # show end


if __name__ == '__main__':
    while True:
        response = requests.request("POST", url)
        if response.text[1] == '1':
            fo = open(config_position, "w")
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

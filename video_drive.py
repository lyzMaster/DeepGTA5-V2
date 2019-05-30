import numpy as np
from keras.models import load_model
from utils.img_process import process, yolo_img_process
import utils.yolo_util as yolo_util
import tensorflow as tf
import cv2
from PIL import Image
import pickle
from deepgtav.messages import frame2numpy
import gzip

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

IMAGE_H, IMAGE_W = 416, 416
CAP_IMG_W, CAP_IMG_H = 1914, 1051

data_path = "video_drive_file/dataset.pz"
data_path = gzip.open(data_path)

def drive(model, image, speed, warning):

    throttle = 0
    breakk = 0

    roi, radar = process(image)

    controls = model.predict([np.array([roi]), np.array([radar]), np.array([speed])], batch_size=1)
    controls = controls[0][0]*5/3.14

    if warning:
        return "--> %lf.2  throttle:%lf  brake:%lf" % (controls, False, True)

    if speed < 5:       # control speed
        throttle = 1
    elif speed < 20:
        throttle = 0.5
    elif speed > 25:
        throttle = 0.0
        breakk = 0.4

    if controls > 0:
        controls = controls
        info = "--> %lf.2  throttle:%lf  brake:%lf" % (controls, throttle, breakk)
    else:
        info = "<-- %lf.2  throttle:%lf  brake:%lf" % (controls, throttle, breakk)

    print(info)

    return info


def main():
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

            try:
                data_dict = pickle.load(data_path)   # 读取数据中的每一帧
                speed = data_dict['speed']
                frame = data_dict['frame']
                frame = frame2numpy(frame,(CAP_IMG_W,CAP_IMG_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            except EOFError:
                print("===========end=============")
                exit(0)

            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(yolo_img_process(frame), axis=0)})
            boxes, scores, labels = yolo_util.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.1)
            image, warning = yolo_util.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

            info = drive(model=model, image=frame, speed=speed, warning=warning)

            result = np.asarray(image)
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            while True:
                cv2.imshow("result", result)
                if cv2.waitKey(0) & 0xFF == 32:   # 点击空格下一张
                    break
                elif cv2.waitKey(0) & 0xFF == ord('q'):  # 点击q退出程序
                    print("====================done===================")
                    exit(0)


if __name__ == '__main__':
    main()

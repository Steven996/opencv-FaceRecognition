import os

import cv2
import sys
import numpy as np
import collections
from PIL import Image, ImageDraw, ImageFont
from face_recognition.train import Model
project_root = os.path.abspath(os.path.dirname(__file__))
def cv2ImgAddText(img, text, left, top, textColor=(255, 255, 255), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def start():
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # 加载模型
    model = Model()
    model.load_model(file_path=os.path.join(project_root, 'resource', 'model', 'face_recognition.model.h5'))

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 人脸识别分类器本地存储路径
    cascade_path = os.path.join(project_root, 'resource', 'haarcascade_frontalface_alt.xml')

    face_list = []
    # 循环检测识别人脸

    num = 0
    #while True:
    for i in range(0, 100):
        ret, frame = cap.read()  # 读取一帧视频

        if ret is True:

            # 图像灰化，降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faceID = model.face_predict(image)
                if faceID >= 0:
                    num = num + 1
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                with open(os.path.join(project_root, 'resource', 'dir_dic.txt'), 'r', encoding="gbk") as f:  # 读文件
                    line = f.read()
                    dic = eval(line)
                face_name = get_keys(dic, faceID)
                face_name = face_name[0] if len(face_name) > 0 else ""
                print(face_name)
                face_list.append(face_name)

                # 文字提示是谁
                frame = cv2ImgAddText(frame, face_name, x + 50, y + 20)
        cv2.imshow("CaptureFace", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if (k & 0xFF == ord('q')) | num >= 10:
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

    res = collections.Counter(face_list)
    final_face_name = res.most_common(1)[0][0]
    print(res)
    if final_face_name == '' and len(dict(res)) > 1:
        if res.most_common(2)[1][1] < 10:
            return 0
        final_face_name = res.most_common(2)[1][0]
    print(final_face_name)
    return final_face_name

if __name__ == '__main__':
    start()

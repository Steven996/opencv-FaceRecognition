import os
import sys
import numpy as np
import cv2
from sklearn.decomposition import PCA

IMAGE_SIZE = 100
project_root = os.path.abspath(os.path.dirname(__file__))
# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h = image.shape[0]
    w = image.shape[1]
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    image = cv2.resize(constant, (height, width))
    image = image[20:100, 10:90]

    #pca = PCA(n_components=50)
    #image = pca.fit_transform(image)
    #image = cv2.transpose(image)
    #image = pca.fit_transform(image)
    #cv2.imwrite('resource/image.jpg', image)
    # 调整图像大小并返回
    return image

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv_img

# 读取训练数据
images = []
labels = []
dir_dic = {}

def read_path(path_name, dir_dic_item = 0):
    for dir_item in os.listdir(path_name):
        global dir_dic
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            dir_dic[dir_item] = dir_dic_item
            dir_dic_item = dir_dic_item + 1
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv_imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                labels.append(path_name)
    with open(os.path.join(project_root, 'resource', 'dir_dic.txt'), 'w', encoding='gbk') as f:  # 写文件
        f.write(str(dir_dic))
    return images, labels


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    # 我和闺女两个人共1200张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
    # 图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)
    with open(os.path.join(project_root, 'resource', 'dir_dic.txt'), 'r', encoding='gbk') as f:  # 读文件
        line = f.read()
        dic = eval(line)
    # 标注数据
    abspath = os.path.abspath(path_name)
    labels = np.array([dic[label[len(abspath)+1:]] for label in labels])
    return images, labels




if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_dataset(os.path.join(project_root, 'resource', 'data'))
    #print(images)
    print(type(images))
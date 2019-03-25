#!/usr/bin/python
# -*- coding: UTF-8 -*-

#   data-analysis
#   mnist.py

"""
    Description: 手写识别
    author : zhangjingbo 
    Date:    2019/3/17
"""
import struct
import matplotlib.pyplot as plt
import numpy as np


class LoadMnist:
    @staticmethod
    def decode_idx3_ubyte(idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:{}, 图片数量: {}张, 图片大小: {}*{}'.format(magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, num_rows, num_cols))
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 {} 张'.format(i + 1))
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)
        return images

    @staticmethod
    def decode_idx1_ubyte(idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:{}, 图片数量: {}张'.format(magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 {} 张'.format(i + 1))
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels


def run():
    train_images_idx3_ubyte_file = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/ml/neturalnetwork/train-images-idx3-ubyte'
    train_labels_idx1_ubyte_file = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/ml/neturalnetwork/train-labels-idx1-ubyte'
    test_images_idx3_ubyte_file = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/ml/neturalnetwork/t10k-images-idx3-ubyte'
    test_labels_idx1_ubyte_file = '/Users/zhangjingbo/Workspaces/PycharmProjects/data-analysis/ml/neturalnetwork/t10k-labels-idx1-ubyte'

    train_images = LoadMnist.decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = LoadMnist.decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = LoadMnist.decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = LoadMnist.decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()
    print('done')
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    run()

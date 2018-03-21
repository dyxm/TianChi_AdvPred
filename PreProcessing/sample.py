# Created by Yuexiong Ding on 2018/3/18.
# 对样本进行欠采样
# 
import os
import pandas as pd
import numpy as np
import random


def read_file(path):
    """读取训练数据"""
    data = pd.read_csv(path, sep=' ')
    return data


def write_to_file(data, path, file_name):
    """保存文件"""
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=False, sep=' ')
    return


def main():
    # 采样次数
    sample_num = 20000
    # 训练数据路径
    train_data_path = r'../DataSet/TrainData/train_001.csv'
    # label数据路径
    label_data_path = r'../DataSet/TrainData/label_001.csv'
    # 数据处理完存放的位置
    save_train_data_path = r'../DataSet/TrainData/'

    print('读取数据...')
    train_data = np.array(read_file(train_data_path))
    label_data = np.array(read_file(label_data_path))
    print('数据读取完毕...')

    new_train_data = []
    new_label_data = []
    # 抽取全部正例
    print('抽取全部正例...')
    for i in range(len(label_data)):
        if label_data[i][0] == 1:
            new_train_data.append(train_data[i])
            new_label_data.append(label_data[i])

    # 对负样本进行抽样
    print('对负样本进行抽样...')
    for i in range(len(label_data)):
        if len(new_train_data) >= sample_num:
            break
        if label_data[i][0] == 0:
            if random.randint(1, 100) < 4:
                new_train_data.append(train_data[i])
                new_label_data.append(label_data[i])

    # 结合两组数据
    all_data = np.c_[new_label_data, new_train_data]
    # 打乱样本顺序
    print('打乱样本...')
    all_data = pd.DataFrame(all_data)
    all_data = all_data.sample(frac=1)

    # 保存文件
    # label文件
    print('保存label文件...')
    write_to_file(all_data.iloc[:, 0: 1], save_train_data_path, r'label_balance_001.csv')
    # train文件
    print('保存train文件...')
    write_to_file(all_data.iloc[:, 1:], save_train_data_path, r'train_balance_001.csv')


if __name__ == '__main__':
    main()






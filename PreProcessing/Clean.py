# Created by [Yuexiong Ding] at 2018/3/14
# 数据清理
#

import os
import numpy as np
import pandas as pd


def get_time_feature(datetime):
    """提取时间特征"""
    print('提取时间特征...')
    # month = [i.month for i in datetime]
    day = [i.day for i in datetime]
    hour = [i.hour for i in datetime]
    # return month, day, hour
    return day, hour


def fill(data, method='mode'):
    """用众数对缺失值（-1）进行填充"""
    for i in range(len(data.columns)):
        # 取出某列的众数
        print('替换第 %d 列的缺失值' % i)
        col_data = data.iloc[:, i]
        data.iloc[:, i] = col_data.replace(-1, col_data.mode()[0])

    return data


def normalize(data):
    """归一化"""
    print('归一化数据...')
    norm_data = (data - data.min()) / (data.max() - data.min())
    return norm_data


def write_to_file(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=False, sep=' ')
    return


def main():
    # 原始训练数据路径
    raw_train_data_path = r'../DataSet/TrainData/raw_train_20180301.txt'
    # 数据处理完存放的位置
    save_train_data_path = r'../DataSet/TrainData/'
    # 原始测试数据路径
    # raw_test_data_path = r'../DataSet/TestData/raw_test_a_20180301.txt'

    # 读取数据
    raw_data = pd.read_csv(raw_train_data_path, sep=" ",
                           usecols=['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                                    'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                                    'context_timestamp', 'context_page_id', 'shop_review_num_level',
                                    'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
                                    'shop_score_delivery', 'shop_score_description', 'is_trade'],
                           parse_dates=['context_timestamp'],
                           date_parser=lambda dates: pd.to_datetime(dates, unit='s'))

    # 填充缺失值
    raw_data = fill(raw_data)

    # 将标记写入label.csv文件
    write_to_file(raw_data['is_trade'], save_train_data_path, r'label_001.csv')
    # 删除 is_trade 列
    raw_data = raw_data.drop(['is_trade'], axis=1)

    # 提取时间特征
    # raw_data['month'], raw_data['day'], raw_data['hour'] = get_time_feature(raw_data['context_timestamp'])
    raw_data['day'], raw_data['hour'] = get_time_feature(raw_data['context_timestamp'])
    # 删除 context_timestamp 列
    raw_data = raw_data.drop(['context_timestamp'], axis=1)

    # 归一化
    raw_data = normalize(raw_data)

    # 将特征写入 train.csv 文件
    write_to_file(raw_data, save_train_data_path, r'train_001.csv')

    print(raw_data)


if __name__ == '__main__':
    main()

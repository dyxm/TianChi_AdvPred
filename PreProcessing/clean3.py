# Created by Yuexiong Ding on 2018/3/21.
# 数据清理3
# 
import os
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics


def get_category_property_feature(data):
    for i in range(3):
        data['category_%d' % i] = data['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_category_list']

    print('item_property_list_ing')
    for i in range(3):
        data['property_%d' % i] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_property_list']

    for i in range(3):
        data['predict_category_%d' % i] = data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )
    del data['predict_category_property']


def get_time_feature(data):
    """提取时间特征"""
    print('提取时间特征...')
    data['week'] = [i.weekday for i in data['context_timestamp']]
    data['day'] = [i.day for i in data['context_timestamp']]
    data['hour'] = [i.hour for i in data['context_timestamp']]

    # 用户当天的查询次数
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])

    # 用户当天某个小时的查询次数
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

    return data


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


def write_to_file(data, path, file_name, header=False):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=header, sep=' ')
    return


def main():
    # 是否是训练集
    is_train = True

    # 原始数据路径
    if is_train:
        raw_data_path = r'../DataSet/TrainData/raw_train_20180301.txt'
        save_data_path = r'../DataSet/TrainData/'
        save_file_name = r'train_002.csv'
        usecols = ['instance_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                   'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                   'context_timestamp', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate',
                   'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade']
    else:
        raw_data_path = r'../DataSet/TestData/raw_test_a_20180301.txt'
        save_data_path = r'../DataSet/TestData/'
        save_file_name = r'test_002.csv'
        usecols = ['instance_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                   'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                   'context_timestamp', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate',
                   'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

    # 读取数据
    raw_data = pd.read_csv(raw_data_path, sep=" ", parse_dates=['context_timestamp'], usecols=usecols,
                           date_parser=lambda dates: pd.to_datetime(dates, unit='s'))
    # 取出索引
    instance_id = raw_data['instance_id']

    # 填充缺失值
    raw_data = fill(raw_data)

    # 提取时间特征
    raw_data = get_time_feature(raw_data)
    raw_data = raw_data.drop(['context_timestamp'], axis=1)
    raw_data = raw_data.drop(['user_id'], axis=1)

    # 归一化
    norm_data = normalize(raw_data)

    # instance_id 处理，如果是训练集则不加 instance_id，测试集则加入 instance_id
    if is_train:
        norm_data = norm_data.drop(['instance_id'], axis=1)
    else:
        norm_data['instance_id'] = instance_id

    # 将特征写入 .csv 文件
    write_to_file(norm_data, save_data_path, save_file_name, header=True)

    print(norm_data)


if __name__ == '__main__':
    main()

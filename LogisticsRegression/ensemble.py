# Created by Yuexiong Ding on 2018/3/19.
# 均值集成多个LR
# 
import os
import pandas as pd
import numpy as np


def read_file(path):
    """读取训练数据"""
    data = pd.read_csv(path, sep=' ')
    return data


def write_to_file(data, path, file_name, header=False):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=header, sep=' ', line_terminator='\n')
    return


def main():
    # 预测结果保存路径
    save_predict_path = r'../DataSet/Predict/LR/'

    # 求多个模型的预测值之和
    print('求多个模型的预测值之和...')
    n = 10
    final_predict = read_file(r'../DataSet/Predict/LR/LR_predict_0.csv')
    total_sum = final_predict['predicted_score']
    for i in range(1, n):
        total_sum += read_file(r'../DataSet/Predict/LR/LR_predict_' + str(i) + r'.csv')['predicted_score']

    # 求评均
    print('求多个模型的预测值的均值...')
    final_predict['predicted_score'] = total_sum / n

    # 保存最终集成结果
    print('保存最终集成结果...')
    write_to_file(final_predict, save_predict_path, r'ensemble_predict.csv', header=True)


if __name__ == '__main__':
    main()





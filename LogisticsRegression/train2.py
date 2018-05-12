# Created by Yuexiong Ding on 2018/3/19.
# LR回归2
#
import os
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics


def read_file(path):
    """读取训练数据"""
    print('读取数据...')
    data = pd.read_csv(path, sep=' ')
    return data


def train(train_x, train_y, class_weight):
    """训练模型并返回分类器"""
    print('训练LR模型...')
    classifier = LogisticRegression(class_weight=class_weight, solver='sag', C=10)
    classifier.fit(train_x, train_y)
    return classifier


def predict(classifier, test_x):
    """预测并返回预测类别结果与概率结果"""
    print('预测...')
    predictions = classifier.predict(test_x)
    predictions_proba = classifier.predict_proba(test_x)
    return predictions, predictions_proba


def accuracy(y_true, y_pred):
    """准确率评估"""
    return metrics.accuracy_score(y_true, y_pred)


def precision_score(y_true, y_pred, average):
    """求宏/微平均"""
    return metrics.precision_score(y_true, y_pred, average=average)


def recall(y_true, y_pred, average):
    """求宏/微召回率"""
    return metrics.recall_score(y_true, y_pred, average=average)


def write_to_file(data, path, file_name, header=False):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=header, sep=' ', line_terminator='\n')
    return


def main():
    # 是否是训练
    is_train = False

    # 训练数据路径
    train_data_path = r'../DataSet/TrainData/train_002.csv'

    # 测试的数据路径
    test_data_path = r'../DataSet/TestData/test_002.csv'

    # 预测结果保存路径
    save_predict_path = r'../DataSet/Predict/LR/Train2/'

    # 读取数据
    if is_train:
        # 分训练与测试集
        raw_data = read_file(train_data_path)
        train_data = raw_data.loc[raw_data.day < 1.]  # 18,19,20,21,22,23为训练集
        train_y = np.array(train_data['is_trade']).reshape(1, len(train_data['is_trade']))[0]
        train_x = np.array(train_data.drop(['is_trade'], axis=1))
        test_data = raw_data.loc[raw_data.day == 1.]  # 用第24天作为验证集
        test_y = np.array(test_data['is_trade']).reshape(1, len(test_data['is_trade']))[0]
        test_x = np.array(test_data.drop(['is_trade'], axis=1))
    else:
        train_data = read_file(train_data_path)
        train_y = np.array(train_data['is_trade']).reshape(1, len(train_data['is_trade']))[0]
        train_x = np.array(train_data.drop(['is_trade'], axis=1))
        test_x = read_file(test_data_path)

    # 训练模型
    weight_0 = 0.5
    class_weight = {0: weight_0, 1: 1 - weight_0}
    classifier = train(train_x, train_y, class_weight)

    if is_train:
        # 预测
        predictions, predictions_proba = predict(classifier, test_x)
        positive_count = 0
        for i in range(len(predictions)):
            if predictions[i] == 1:
                positive_count += 1
        print('预测的正例有 %d 个' % positive_count)

        # 模型评估
        print('正确率：%f' % accuracy(test_y, predictions))
        print('宏平均：%f' % precision_score(test_y, predictions, 'macro'))
        print('微平均：%f' % precision_score(test_y, predictions, 'micro'))
        print('宏召回率：%f' % recall(test_y, predictions, 'macro'))
        print('微召回率：%f' % recall(test_y, predictions, 'micro'))
        print('f1：%f' % metrics.f1_score(test_y, predictions, average='weighted'))
        print('loss：%f' % metrics.log_loss(test_y, predictions_proba))
    else:
        # 预测测试集
        print('预测测试集...')
        predictions_proba = classifier.predict_proba(test_x.iloc[:, 1:])
        # print(predictions_proba)
        result = pd.DataFrame(columns=['instance_id'], data=np.array(test_x.iloc[:, 0: 1]))
        result['predicted_score'] = pd.DataFrame(predictions_proba[:, 1:])

        # 保存预测结果
        print('保存预测结果...')
        write_to_file(result, save_predict_path, r'LR_predict_2.csv', header=True)


if __name__ == '__main__':
    main()

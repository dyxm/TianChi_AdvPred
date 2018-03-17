# Created by [Yuexiong Ding] at 2018/3/16
# logistic regression
#
import numpy as np
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
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


def loss(y_true, y_pred):
    m = len(y_true)
    y_true = y_true.reshape(m, 1)
    cost = (y_true * np.log(y_pred[:, 1: 2]) + (1 - y_true) * np.log(y_pred[:, 0: 1])).sum()
    return - cost / m


def main():
    # 训练数据路径
    train_data_path = r'../DataSet/TrainData/train_001.csv'
    # label数据路径
    label_data_path = r'../DataSet/TrainData/label_001.csv'

    # 读取数据
    train_data = read_file(train_data_path)
    label_data = read_file(label_data_path)
    N = 10
    for j in range(N):
        train_x, test_x, train_y, test_y = train_test_split(train_data, label_data, test_size=0.1)
        train_y = np.array(train_y).reshape(1, len(train_y))[0]
        test_y = np.array(test_y).reshape(1, len(test_y))[0]

        # 训练模型
        weight_0 = 0.05
        class_weight = {0: weight_0, 1: 1 - weight_0}
        classifier = train(np.array(train_x), np.array(train_y).reshape(1, len(train_y))[0], class_weight)

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
        print('loss：%f' % loss(test_y, predictions_proba))


if __name__ == '__main__':
    main()

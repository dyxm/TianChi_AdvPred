# Created by [Yuexiong Ding] at 2018/3/17
# 多隐层神经网络--预测
#

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf


def write_to_file(data, path, file_name, header=False):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(path + file_name, index=False, header=header, sep=' ', line_terminator='\n')
    return


def read_file(path):
    """读取训练数据"""
    data = pd.read_csv(path, sep=' ')
    return data


def add_layer(layer_name, inputs, in_size, out_size, activation_function=None, drop_out=None):
    """添加隐层并返回该层运算结果"""
    with tf.variable_scope(layer_name, reuse=None):
        weights = tf.get_variable("weights", shape=[in_size, out_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1, out_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

    w_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = w_plus_b
    else:
        outputs = activation_function(w_plus_b)
    if drop_out is not None:
        outputs = tf.nn.dropout(outputs, drop_out)
    return outputs


def main():
    # 测试数据路径
    test_data_path = r'../DataSet/TestData/test_002.csv'
    # 预测结果保存路径
    save_predict_path = r'../DataSet/Predict/NN/Model1/'
    # 模型存储地址
    CKPT_PATH = '../Ckpt_Dir/AdamOptimizer_17_100_150_100_50_2'
    # drop_out因子
    drop_out = 1

    # 输入
    x = tf.placeholder(tf.float32, [None, 19])

    # 第一层--输入层
    l1 = add_layer("input", x, 19, 100, activation_function=tf.nn.relu)
    # 第二层
    l2 = add_layer("layer2", l1, 100, 150, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第三层
    l3 = add_layer("layer3", l2, 150, 100, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第四层
    l4 = add_layer("layer4", l3, 100, 50, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第五层
    l5 = add_layer("layer5", l4, 50, 50, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第六层--输出层
    prediction = add_layer("output", l5, 50, 1, activation_function=tf.sigmoid)

    # 创建存储模型路径
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    saver = tf.train.Saver()

    # 打开会话
    with tf.Session() as sess:

        # 读取数据
        print('读取数据...')
        test_data = read_file(test_data_path)
        result = pd.DataFrame(columns=['instance_id'], data=np.array(test_data.instance_id))
        test_x = np.array(test_data.drop(['instance_id'], axis=1))

        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 获取模型检查点
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 开始预测
        y_pred = sess.run(prediction, feed_dict={x: test_x})
        result['predicted_score'] = y_pred

        # 保存预测结果
        print('保存预测结果...')
        write_to_file(result, save_predict_path, r'NN_predict_1.csv', header=True)
        print('成功')


if __name__ == '__main__':
    main()

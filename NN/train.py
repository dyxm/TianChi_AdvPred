# Created by [Yuexiong Ding] at 2018/3/17
# 多隐层神经网络
#

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
    # 训练数据路径
    train_data_path = r'../DataSet/TrainData/train_001.csv'
    # label数据路径
    label_data_path = r'../DataSet/TrainData/label_001.csv'
    # 训练次数
    iterator = 10000
    # batch 大小
    batch_size = 10000
    # 模型存储地址
    CKPT_PATH = '../Ckpt_Dir/AdamOptimizer_17_100_150_100_50_2'
    # drop_out因子
    drop_out = 0.7
    # 代价敏感因子
    w_ls = tf.Variable([[0.055], [10]], name="w_ls", trainable=False)
    # 计数器变量，当前第几步
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 输入
    x = tf.placeholder(tf.float32, [None, 17])
    # 输出
    y = tf.placeholder(tf.float32, [None, 1])

    # 第一层--输入层
    l1 = add_layer("input", x, 17, 100, activation_function=tf.nn.relu)
    # 第二层
    l2 = add_layer("layer2", l1, 100, 150, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第三层
    l3 = add_layer("layer3", l2, 150, 100, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第四层
    l4 = add_layer("layer4", l3, 100, 50, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第五层
    l5 = add_layer("layer5", l4, 50, 50, activation_function=tf.nn.relu, drop_out=drop_out)
    # 第六层--输出层
    prediction = add_layer("output", l5, 50, 1, activation_function=tf.tanh)
    # logistic 误差
    # loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction,1e-10,1.0)), reduction_indices=1))
    loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(prediction), reduction_indices=1))
    # Adam优化器
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # 计算模型当前的正确率
    # correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建存储模型路径
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    saver = tf.train.Saver()

    # 打开会话
    with tf.Session() as sess:
        # 读取数据
        print('读取数据...')
        train_data = read_file(train_data_path)
        label_data = read_file(label_data_path)
        # 划分数据
        train_x, test_x, train_y, test_y = train_test_split(train_data, label_data, test_size=0.1)

        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 获取模型检查点
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        global_start_step = global_step.eval()
        print("从 %d 步开始训练..." % global_start_step)

        # 开始训练
        for i in range(global_start_step, iterator):
            j = 0
            for (start, end) in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x) + 1, batch_size)):
                j += 1
                batch_x = train_x[start: end]
                batch_y = train_y[start: end]
                # 训练
                print('第 %d 次训练第 %d 个batch...' % (i, j))
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            # 训练 10 个回合用测试集计算并输出损失
            if i % 10 == 0:
                # 计算所有数据的交叉熵损失
                total_loss = sess.run(loss, feed_dict={x: test_x, y: test_y})
                # 输出交叉熵之和
                print("训练 %d 次后测试集交叉熵损失为：%f " % (i, total_loss))

        # 更新当前步数
        global_step.assign(i).eval()
        # 保存模型
        saver.save(sess, CKPT_PATH + "/model.ckpt", global_step=global_step)


if __name__ == '__main__':
    main()

from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt



# tf.device('/gpu:0')
# tensorflow 实现 Logistic Regression
# 读取数据
x_test = pd.read_csv("testdatasetin.csv", header=None)  # 测试集特征
x_train = pd.read_csv("traindatasetin.csv", header=None)  # 训练集特征
y_test = pd.read_csv("testdatasetout.csv", header=None)  # 测试集标签
y_train = pd.read_csv("traindatasetout.csv", header=None)  # 训练集标签


y_train = tf.concat([1 - y_train, y_train], 1)
y_test = tf.concat([1 - y_test, y_test], 1)


# 参数定义
learning_rate = 0.05  # 学习率
training_epochs = 180  # 训练迭代次数
batch_size = 100  # 分页的每页大小（后面训练采用了批量处理的方法）
display_step = 10  # 何时打印到屏幕的参量

n_samples = x_train.shape[0]  # sample_num 训练样本数量
n_features = x_train.shape[1]  # feature_num 特征数量 256
n_class = 2
# 变量定义
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_class])
# 权重定义
W = tf.Variable(tf.zeros([n_features, n_class]), name="weight")
b = tf.Variable(tf.zeros([n_class]), name="bias")

# y=x*w+b 线性
pred = tf.matmul(x, W) + b

# 准确率
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 损失
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化
init = tf.global_variables_initializer()


train_accuracy = []
test_accuracy = []
avg_cost = []
# 训练
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            _, c = sess.run([optimizer, cost],
                            feed_dict={x: x_train[i * batch_size: (i + 1) * batch_size],
                                       y: y_train[i * batch_size: (i + 1) * batch_size, :].eval()})

            train_accuracy.append(accuracy.eval(
                {x: x_train, y: y_train.eval()}))
            test_accuracy.append(accuracy.eval(
                {x: x_test, y: y_test.eval()}))
            avg_cost.append(c / total_batch)

        if (epoch + 1) % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", c / total_batch)

    print("Optimization Finished!")
    print("Testing Accuracy:", accuracy.eval(
        {x: x_test, y: y_test.eval()}))
    #print(sess.run(W))
    #print(sess.run(b))

    plt.suptitle("learning rate=%f  training epochs=%i  sample_num=%i" % (
        learning_rate, training_epochs, n_samples), size=14)
    plt.plot(avg_cost)
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(['loss', 'train_accuracy', 'test_accuracy'])
    plt.ylim(0., 1.5)
    plt.xlabel("Epochs")
    plt.ylabel("Rate")
    plt.show()

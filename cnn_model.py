# coding: utf-8

import tensorflow as tf

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 128  # 词向量维度
    seq_length = 4000  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 512  # 卷积核数目
    kernel_sizes = [2, 3, 5]  # 卷积核尺寸
    vocab_size = 10000  # 词汇表大小，修改此参数需要。。。。。。。。。

    hidden_dim = 1024  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-4  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据，None为batch_size
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        # Tensorflow默认是在GPU执行运算，但是embedding的实现不支持GPU。用tf.device("/cpu:0")强制在cpu执行运算。
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            # tf.nn.embedding_lookup返回一个三维张量[batch_size, seq_length, embedding_dim]
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_outputs = []
        for kernel_size in self.config.kernel_sizes:
            with tf.name_scope("cnn-maxpool-%s" % kernel_size):
                # 卷积层
                # tf.layers.conv1d除了输入数据只有两个必要参数：滤波器数量和卷积核的长度，卷积核的宽与词向量的维度一样
                # 返回一个三维张量[batch_size, seq_length-kernel_size+1, num_filters]
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, kernel_size, name='conv-%s' % kernel_size)
                # 有人在这里加一层relu激活
                # conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大池化层
                # 由于reduction_indices=[1]，是在seq_length-kernel_size+1的维度上取最大值
                # 返回一个二维张量[batch_size, num_filters]
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp-%s' % kernel_size)
                pooled_outputs.append(gmp)

        # 拼接每种滤波器卷积以及池化之后的结果
        self.h_pool = tf.concat(pooled_outputs, 1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # tf.layers.dense返回二维张量[batch_size, hidden_dim]
            fc = tf.layers.dense(self.h_pool, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc) # 返回还是二维张量[batch_size, hidden_dim]

            # 分类器
            # 还是全连接层，返回二维张量[batch_size, num_classes]
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # 预测类别，最大的下标就是预测类别，返回一维张量维度为batch_size
            self.y_pred_cls = tf.argmax(self.logits, 1)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

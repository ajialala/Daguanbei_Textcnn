# DC平台 达观杯智能文本分类 Textcnn模型
代码参考：https://github.com/gaussic/text-classification-cnn-rnn
## 环境
- python3
- tensorflow 1.2.1
- anaconda3
## 数据集
将官方的train_set.csv放到data文件夹下，运行data文件夹下的prepare_data.py。将原数据集中的id与article去掉。然后按照9:0.5:0.5的比例分割为训练集，验证集和测试集。
## 原理
![textcnn原理图](pic/Textcnn.png)
*from:Zhang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification*
一般滤波器的宽和词向量的长度相同，高度是可变的，一般情况下是2-5。
## 未完。。。
## 参考链接
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

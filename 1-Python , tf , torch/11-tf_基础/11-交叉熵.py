""""""
import tensorflow as tf

# %% sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)：计算交叉熵
'''
    logits：假设神经网络最后一层的输出为长度为v的向量，如果有batch的话，则logits大小就是[batchsize，v]
    labels：比如两个样本的真实标签分别为2和0，则lables是向量[2,0]
'''
predict_logits = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])  # 假设模型对两个单词预测时，产生的logit分别是[2.0, -1.0, 3.0]和[1.0, 0.0, -0.5]
sess = tf.Session()

word_labels = tf.constant([2, 0])  # 假设词汇表的大小为3， 语料包含两个单词"2 0"   <---
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)
print(sess.run(loss))  # [ 0.32656264  0.46436879]  对应两个预测的perplexity损失。

# %% softmax_cross_entropy_with_logits(logits, labels, name=None)：与上面的函数相似，但是需要将预测目标以概率分布的形式给出。
'''
    logits：神经网络最后一层的输出为必须是长度为num_classes的向量，如果有batch的话，[batchsize，num_classes]，单样本的话，大小就是num_classes
    labels：比如类别总数为3，有两个样本的真实标签分别为2和0，则lables=[ [0.0, 0.0, 1.0], [1.0, 0.1, 0.0] ]    
'''
word_prob_distribution = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # 假设词汇表的大小为3， 语料包含两个单词"2 0"  <---
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logits)
print(sess.run(loss))  # 与上面相同：[ 0.32656264,  0.46436879]

# %% label smoothing：将正确数据的概率设为一个比1.0略小的值，将错误数据的概率设为比0.0略大的值，这样可以避免模型与数据过拟合，在某些时候可以提高训练效果。
word_prob_smooth = tf.constant([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_smooth, logits=predict_logits)
print(sess.run(loss))  # \ [ 0.37656265,  0.48936883]

# %% tf.one_hot(*,num)
indices = [0, 1, 2]  # \ indices = tf.constant([1, 2, 3])
result = tf.one_hot(indices, 3)
sess = tf.Session()
print(sess.run(result))

# [[ 0.  1.  0.]
#  [ 0.  0.  1.]
#  [ 0.  0.  0.]]

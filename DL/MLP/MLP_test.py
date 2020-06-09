import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
""""
784个输入层神经元——300个隐层神经元——10个输出层神经元
"""

'''导入MNIST手写数据'''
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

'''自定义神经层添加函数'''

def add_layer(inputs, in_size, out_size, activation_function=None):

    '''定义权重项'''

    '''tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差
    这个函数产生正太分布，均值和标准差自己设定'''
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0, stddev=0.2))

    '''定义bias项'''
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    '''权重与输入值的矩阵乘法+bias项'''
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    '''根据激活函数的设置来处理输出项'''
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

'''创建样本数据和 dropout参数的输入部件
dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
dropout :使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小！
'''

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

'''利用自编函数来生成隐层的计算部件，这里activation_function设置为relu'''
l1 = add_layer(x, 784, 300, activation_function=tf.nn.relu)

'''对l1进行dropout处理'''
l1_dropout = tf.nn.dropout(l1, keep_prob)

'''利用自编函数对dropout后的隐层输出到输出层输出创建计算部件，这里将activation_function改成softmax
softmax: 数据结果以概率表达
'''
prediction = add_layer(l1_dropout, 300, 10, activation_function=tf.nn.softmax)

'''根据均方误差构造loss function计算部件'''

loss = tf.reduce_mean(tf.reduce_sum((prediction - y)**2, reduction_indices=[1]))

'''定义优化器部件，AdagradOptimizer优化器'''

train_step = tf.train.AdagradOptimizer(learning_rate = 0.3).minimize(loss)

'''激活所有部件'''

init = tf.global_variables_initializer()

'''创建新会话'''
sess = tf.Session()

'''在新会话中激活所有部件'''
sess.run(init)

'''10001次迭代训练，每200次输出一次当前网络在测试集上的精度'''
for i in range(10001):
    '''每次从训练集中抽出批量为200的训练批进行训练'''
    x_batch, y_batch = mnist.train.next_batch(200)
    '''激活sess中的train_step部件'''
    sess.run(train_step, feed_dict={x:x_batch, y:y_batch, keep_prob:0.75})
    '''每200次迭代打印一次当前精度'''
    if i % 200 == 0:
        print('第',i,'轮迭代后：')
        '''创建精度计算部件'''
        whether_correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

        accuracy = tf.reduce_mean(tf.cast(whether_correct, tf.float32))  # 参数类型转换

        '''在sess中激活精度计算部件来计算当前网络的精度'''
        print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                            y:mnist.test.labels,
                                            keep_prob:1.0}))
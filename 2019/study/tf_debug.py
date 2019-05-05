"""
tensorflow几种常用调试方式如下:
1.通过Session.run()获取变量的值
2.利用Tensorboard查看一些可视化统计
3.使用tf.Print()和tf.Assert()打印变量
4.使用Python的debug工具: ipdb, pudb
5.利用tf.py_func()向图中插入自定义的打印代码, tdb
6.使用官方debug工具: tfdbg

tensorflow是通过先建图再运行的方式进行运行,这就使得我们写在图建立过程中的输出语句在图运行的时候并不能得到执行,从而使得调试困难. 
我们想在运行过程中,对训练的一些变量进行追踪和打印,对一些错误进行输出分析,下面介绍几种在tensorflow中进行debug的方法.

"""

import tensorflow as tf 

# 1 using Session.run()获取变量的值
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
bias = tf.Variable(1.0)

y_pred = x ** 2 + bias 
loss = (y - y_pred) ** 2 
sess = tf.Session() 

# Error: to compute loss, y is required as a dependency 
# print("Loss(x,y) = %.3f" % sess.run(loss, {x: 3.0}))



# 2 using tf.Print 
import tensorflow.contrib as tc 
def multilayer_perceptron(x):
    fc1 = tc.layers.fully_connected(x, 256, activation_fn = tf.nn.relu)
    fc2 = tc.layers.fully_connected(fc1, 256, activation_fn = tf.nn.relu)
    out = tc.layers.fully_connected(fc2, 10, activation_fn = None)
    out = tf.Print(out, [tf.argmax(out, 1)],
                'argmax(out)= ', summarize=20, first_n=7)
    return out 

    



a = tf.random_normal(shape=[4,3])
a = tf.Print(a, [a], message="This is a: ")

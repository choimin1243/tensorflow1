
import numpy as np

import tensorflow as tf


srcData=np.array([[73,80,152],[93,88,185],[89,91,180],[96,98,196],[76,66,142]])

data=np.array(srcData[:,0:-1],dtype=float)
label=np.array(srcData[:,[-1]],dtype=float)
print(data.shape)

X=tf.placeholder(tf.float32,shape=[None,2],name="data")
Y=tf.placeholder(tf.float32,shape=[None,1],name="label")
W=tf.Variable(tf.random_normal([2,1],name="weight"))
b=tf.Variable(tf.random_normal([1]),name="bias")
hypothesis=tf.matmul(X,W)+b
loss=tf.reduce_mean(tf.square(hypothesis-Y+b))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


for step in range(301):
    loss_val, hy_val,_=sess.run([loss,hypothesis,optimizer],feed_dict={X:data,Y:label})
    if step%10==0:
        print(loss_val)
        print("----------------------------")
        print(hy_val)
        print("----------------------------")

print(sess.run(hypothesis,feed_dict={X:[[70,90]]}))



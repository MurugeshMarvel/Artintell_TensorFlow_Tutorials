import tensorflow as tf

#model parameters
w = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#modle input and output
x = tf.placeholder(tf.float32)
linear_model = w *x +b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model - y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [-1,-2,-3,-4]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y:y_train})
#evaluating training accuracy
curr_w , curr_b, curr_loss = sess.run([w, b,loss],{x:x_train,y:y_train})
print("w:%s b:%s loss:%s"%(curr_w,curr_b,curr_loss))

for i in range(10000):
    sess.run(train, {x: x_train, y:y_train})
#evaluating training accuracy
curr_w , curr_b, curr_loss = sess.run([w, b,loss],{x:x_train,y:y_train})
print("w:%s b:%s loss:%s"%(curr_w,curr_b,curr_loss))

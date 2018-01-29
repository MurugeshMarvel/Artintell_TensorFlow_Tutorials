import tensorflow as tf

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

input_data = tf.placeholder(tf.float32)
output_data = tf.placeholder(tf.float32)

linear_model = input_data * w + b

loss = tf.reduce_sum(tf.square(linear_model - output_data))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

inp = [1,2,3,4]
out = [0,-1,-2,-3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train ,  {input_data: inp, output_data:out})

curr_w, curr_b, curr_loss = sess.run([w,b, loss], {input_data: inp, output_data:out})
print ("w : %s b: %s loss: %s"%(curr_w, curr_b, curr_loss))

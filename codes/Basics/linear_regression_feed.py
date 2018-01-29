import tensorflow as tf
tf.set_random_seed(123)

w = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

input_data = tf.placeholder(tf.float32, shape=[None])
output_data = tf.placeholder(tf.float32, shape =None)

func = input_data * w + b

cost = tf.reduce_mean(tf.square(func - output_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    cost_val, w_val, b_val ,_ = sess.run([cost, w, b, train], feed_dict={input_data:[1,2,3], output_data:[10,20,30]})
    if step % 20 == 0:
        pass
        #print step, cost_val, w_val, b_val

print (sess.run(func, feed_dict={input_data:[7]}))
print (sess.run(func, feed_dict={input_data:[2.4, 5.6]}))

print "*"*20
for step in range(2000):
    cost_val , w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={input_data:[1,2,3,4,5], output_data:[2.1, 3.1, 4.1, 5.1,6.1]})
    if step % 20 ==0:
        pass
        #print (step, cost_val, w_val, b_val)

print sess.run(func, feed_dict = {input_data:[7]})
print sess.run(func, feed_dict = {input_data:[6.5, 4.7,9.7]})

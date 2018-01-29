import tensorflow as tf
tf.set_random_seed(345)

input_data = [1,2,3,4,5]
output_data = [10, 20, 30, 40, 50]

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

func = input_data * w + b

cost = tf.reduce_mean(tf.square(func - output_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(w), sess.run(b))
a = 23 * w + b
print a.eval(session=sess)

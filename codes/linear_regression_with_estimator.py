import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01,-4.1,-7,0.])
input_fn = tf.estimator.inputs.numpy_input_fn()

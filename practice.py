# import math
# import random
# import os
# import numpy as np
# import tensorflow as tf
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#
# # 배열에서 [0,0] 없애는 방법 - 저장해 놓기
# x = tf.placeholder(tf.float32, shape=(4, 2))
# zeros_vector = tf.zeros(shape=(1, 4), dtype=tf.float32)
# intermediate_tensor = tf.reduce_sum(tf.abs(x), 1)
# bool_mask = tf.squeeze(tf.not_equal(intermediate_tensor, zeros_vector))
# omit_zeros = tf.boolean_mask(x, bool_mask)
#
# init = tf.global_variables_initializer()
# sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
# sess.run(init)
# feed_dict = {x: np.array([[1, 2], [3, 4], [5, 6], [0, 0]])}
#
# print(sess.run(omit_zeros, feed_dict=feed_dict))
import queue

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity([[1, 0, -1]], [[-1, -1, 0]]))

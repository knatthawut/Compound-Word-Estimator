from gensim.models.keyedvectors import KeyedVectors
import json
from scipy import stats
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sys
import utils

#get input filename
filename = 'word_representation'

#hyper parameter
learning_rate = 0.00005
training_iters = 5000
batch_size = 5000
num_hidden = 128
max_length = 15
# save model to
savePath = "./model/" + filename + ".ckpt"



# Getting Length of RNN Seq
def _length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def prediction(wordList, word_vectors, rnn_model):

    # Define vector range according to input vector
    vec_length = len(word_vectors[wordList[0][0]])


    # Setting Structure for RNN and setup GPU config
    rnn_graph = tf.Graph()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #sess_rnn = tf.Session(graph=rnn_graph, config=tf.ConfigProto(gpu_options=gpu_options))
    sess_rnn = tf.Session(graph=rnn_graph)

    with rnn_graph.as_default():
        #Set Placeholder
        data = tf.placeholder(tf.float32, [None, max_length, vec_length])
        target = tf.placeholder(tf.float32, [None, vec_length])

        #RNN Network
        output, last = rnn.dynamic_rnn(
        rnn_cell.GRUCell(num_hidden),
        data,
        dtype=tf.float32,
        sequence_length=_length(data),
        time_major=False)

        in_size = num_hidden
        out_size = int(target.get_shape()[1])
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        Weight = tf.Variable(weight)
        Bias = tf.Variable(bias)

        # Predict representations from the last word
        # Getting index (Index of the data in Tensor) and Flat tensor and Slice tensor to get last word
        seq_output = tf.shape(output)[0]
        output_size = int(output.get_shape()[2])
        index = tf.range(0, seq_output) * max_length + (_length(data) - 1)
        flat = tf.reshape(output, [-1, output_size])
        last_word = tf.gather(flat, index)

        # Estimated Representation
        prediction = tf.matmul(last_word, Weight) + Bias

        # Loss Function & Optimizer
        loss = tf.reduce_mean(tf.square(target - prediction))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #Loading Model
        saver.restore(sess_rnn, rnn_model)

        # Init dic
        rnn_vec = {}
        #Predicting Vector
        for word in wordList:
            if utils.checkWordExist(word,word_vectors):
                rnn_prediction = sess_rnn.run(prediction, {data: np.array([np.array(utils.getSample(word, word_vectors,max_length, vec_length), dtype=np.float).astype(np.float32)])})
                rnn_vec['_'.join(word)] = rnn_prediction[0]
            else:
                rnn_vec['_'.join(word)] = None

    sess_rnn.close()
    return rnn_vec

# Testing the model
if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format(filename + '.bin', binary=True)
    print('Loading W2V Model Completed')

    # Preparing List of Entities -> [[a,b],[c,d,e]],...]
    #test_data = [['pyridostigmine','bromide'], ['alternating', 'gradient', 'synchrotron']]
    test_data = []
    with open('testing.txt', 'r') as fp:
        for wp in fp:
            words = wp[:-1].split('\t')
            test_data.append(words[0].split())
    print('Load Data Completed')

    # getting estimation vector
    predict_vec = prediction(test_data, word_vectors, savePath)

    #Evaluation
    utils.test(word_vectors,test_data,predict_vec)
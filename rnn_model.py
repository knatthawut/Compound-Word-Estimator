from gensim.models.keyedvectors import KeyedVectors
import json
from scipy import stats
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sys
import utils

#get input filename (word representation files)
filename = 'word_representation'

#hyper parameter
learning_rate = 0.00005
training_iters = 5000
batch_size = 10000
num_hidden = 128
max_length = 8
# save model to
savePath = "./model/" + filename + ".ckpt"


# return the length of data



# Getting Length of RNN Seq
def _length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def train(train_data, word_vectors):

    # Define vector range according to input vector
    vec_length = len(word_vectors[train_data[0][0]])

    # Option Find Max for Padding
    max_length = utils.getMaxToken(train_data)
    print 'Max Len :', str(max_length)

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

        # Number of tuple of one batch
        training_sample= len(train_data)
        print('Train :' + str(training_sample))


        # Start Training
        #Init Param
        sess_rnn.run(init)
        print('------Training RNN Model-----')
        count = 0
        for epoch in range(int(training_iters)):
            for _ in range(10):
                try:
                    train, label = utils.loadData(train_data, word_vectors, (batch_size * count) % training_sample,batch_size,max_length,vec_length)
                    count = count + 1

                    sess_rnn.run([optimizer], {data: train, target: label})

                except Exception as e:
                    print("Error : {0}".format(str(e)).encode("utf-8"))

            # print epoch,count
            test_data, test_label = utils.loadData(train_data, word_vectors, (batch_size * count) % training_sample, batch_size,max_length,vec_length)
            try:
                error = sess_rnn.run(loss, {data: test_data, target: test_label})
            except Exception as e:
                print("Error : {0}".format(str(e)).encode("utf-8"))
            if ((epoch + 1) % 10) == 0:
                print('Epoch {:2d} error {:3.10f} '.format(epoch + 1, error))
        saver.save(sess_rnn, savePath)

    sess_rnn.close()



# Training the model
if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format(filename+'.bin', binary=True)
    print('Loading W2V Model Completed')

    #Option Generate Training data (automatically create training.txt)
    utils.genTrainingFromVocab(word_vectors)
    print('Generated Training Data')

    # Getting pair of entity [a_b, [a,b],...] for training
    train_data = []
    with open('training.txt','r') as fp:
        for wp in fp:
            words = wp[:-1].split('\t')
            train_data.append([words[1],words[0].split()])
    print('Load Data Completed')

    train(train_data, word_vectors)

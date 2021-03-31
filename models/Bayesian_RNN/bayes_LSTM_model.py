import tensorflow as tf

from bayes_LSTMCell import BayesianLSTMCell
from model_utils import variationalPosterior
from tensorflow_probability.python.distributions import Normal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sqlite3
import pandas as pd
import numpy as np

tf.compat.v1.disable_eager_execution()


class SentimentAnalysisMultiLayerLSTM:

    def __init__(self, training):

        self.LSTM_KL=0
        self.embedding_dim = 47  # the number of hidden units in each RNN
        self.keep_prob = 0.5
        self.batch_size = 1
        self.lstm_sizes = [100]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.max_sequence_length = 100
        self.prior=(0,1) #univariator prior
        self.isTraining=training


        with tf.compat.v1.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.compat.v1.compat.v1.placeholder(tf.float32, shape=[None, None, self.embedding_dim])
            self.groundtruths = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes])

        with tf.compat.v1.variable_scope('rnn_cell'):
            self.initial_state, self.final_lstm_outputs, self.final_state, self.cell = self.build_lstm_layers(self.lstm_sizes, self.inputs,self.keep_prob, self.batch_size)



            self.softmax_w, self.softmax_w_mean, self.softmax_w_std=  variationalPosterior((self.lstm_sizes[-1], self.num_classes), "softmax_w", self.prior, self.isTraining)
            self.softmax_b, self.softmax_b_mean, self.softmax_b_std = variationalPosterior((self.num_classes), "softmax_b", self.prior, self.isTraining)
            self.logits=tf.compat.v1.nn.xw_plus_b(self.final_lstm_outputs,  self.softmax_w,self.softmax_b)

        with tf.compat.v1.variable_scope('rnn_loss', reuse=tf.compat.v1.AUTO_REUSE):

            if (self.isTraining):
                self.KL=0.
                # use cross_entropy as class loss
                self.loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.groundtruths, logits=self.logits)
                self.KL=tf.math.add_n(tf.compat.v1.get_collection("KL_layers"), "KL")

            self.cost=(self.loss+self.KL)/self.batch_size  #the total cost need to divide by batch size
            self.optimizer = tf.compat.v1.train.AdamOptimizer(0.02).minimize(self.loss)

        # with tf.compat.v1.variable_scope('rnn_accuracy'):
        #     self.accuracy = tf.compat.v1.metrics.accuracy(labels=tf.math.argmax(self.groundtruths, axis=1), predictions=self.prediction)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())  # don't forget to initial all variables
        self.saver = tf.compat.v1.train.Saver()  # a saver is for saving or restoring your trained weight

        print("Completed creating the graph")

    def train(self, batch_x, batch_y, state):

        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = state
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, final_state, _ = self.sess.run([self.loss, self.accuracy, self.final_state, self.optimizer], fd)

        return loss, accuracy, final_state

    def test(self, batch_x, batch_y, batch_size):

        """
         NEED TO RE-WRITE this function interface by adding the state
        :param batch_x:
        :param batch_y:
        :return
        """
        # restore the model

        # with tf.Session() as sess:
        #    model=model.restore();

        test_state = model.cell.zero_state(batch_size, tf.float32)
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = test_state
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], fd)

        return prediction, accuracy

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/rnn_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/rnn_%d.ckpt' % (e))

    def build_lstm_layers(self, lstm_sizes, inputs, keep_prob_, batch_size):
        """
        Create the LSTM layers
        inputs: array containing size of hidden layer for each lstm,
                input_embedding, for the shape batch_size, sequence_length, emddeding dimension [None, None, 384],
                None and None are to handle variable batch size and variable sequence length
                keep_prob for the dropout and batch_size
        outputs: initial state for the RNN (lstm) : tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]
                 outputs of the RNN [Batch_size, sequence_length, last_hidden_layer_dim]
                 RNN cell: tensorflow implementation of the RNN cell
                 final state: tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]
        """
        self.lstms=[]
        for i in range (0,len(lstm_sizes)):
            self.lstms.append(BayesianLSTMCell(lstm_sizes[i], self.prior, self.isTraining, 'lstm'+str(i)))

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(self.lstms)
        # Getting an initial state of all zeros

        initial_state = cell.zero_state(batch_size, tf.float32)
        # perform dynamic unrolling of the network, for variable
        #lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

        # we avoid dynamic RNN, as this produces while loop errors related to gradient checking
        if True:
            outputs = []
            state = initial_state
            with tf.compat.v1.variable_scope("RNN"):
                for time_step in range(self.max_sequence_length):
                    if time_step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        final_lstm_outputs = cell_output
        final_state = state
        #outputs=tf.reshape(tf.concat(1, outputs), [-1, self.embedding_dim])


        return initial_state, final_lstm_outputs, final_state, cell

def training_data():
    df = pd.read_csv('sample_data.csv')
    return df


def format_data(data):
    '''

    seperates data first by session, then by lap, before padding each array so that
    they are all the same length for model input.
    Performs test train split also

    '''


    session_groups = data.groupby('sessionUID')
    training_data = []
    target_data = []
    total_laps = 0
    for s in list(session_groups.groups):
        session = session_groups.get_group(s)
        lap_groups = session.groupby('currentLapNum')
        total_laps += len(lap_groups)
        for l in list(lap_groups.groups):
            lap = lap_groups.get_group(l)
            lap = lap.drop('sessionUID', axis=1)
            target_data.append(lap.pop('lap_time_remaining'))
            training_data.append(lap)
    training = [x.to_numpy() for x in training_data]
    target = [y.to_numpy() for y in target_data]
    print(f'Total Laps: {total_laps}')

    max_timesteps = max(training, key=len).shape[0]
    num_rows_to_add = [max_timesteps-l.shape[0] for l in training]

    training_pad = []
    target_pad = []
    print(f'max timesteps : {max_timesteps}')

    for i in range(len(training)):
        rows_to_add = num_rows_to_add[i]

        training_arr = training[i]
        training_append = np.zeros((rows_to_add, training[0].shape[1]), dtype=float)
        training_array = np.vstack((training_arr, training_append))
        training_pad.append(training_array)

        target_arr = target[i]
        target_append = np.zeros((rows_to_add), dtype=float)
        target_array = np.concatenate([target_arr, np.zeros(rows_to_add)])
        target_pad.append(target_array)

    split = int(total_laps*0.9)

    X_train = training_pad[:split]
    X_test = training_pad[split:]
    y_train = target_pad[:split]
    y_test = target_pad[split:]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':

    # hyperparameter of our network
    EPOCHS = 20
    tf.compat.v1.reset_default_graph()
    model = SentimentAnalysisMultiLayerLSTM(training=True)

    print('collecting data')
    data = training_data()
    print('data collected')
    X_train, X_test, y_train, y_test = format_data(data)

    timesteps = X_train[0].shape[0]
    features = X_train[0].shape[1]

    n_train = X_train.shape[0]

    print(f'n_train is {n_train}')

    BATCH_SIZE = model.batch_size
    print("BATCH SIZE : " + str(BATCH_SIZE))

    rec_loss = []


    for epoch in range(EPOCHS):

        state = model.sess.run([model.initial_state])
        loss_train = 0
        accuracy_train = 0

        for idx in range(0, n_train, BATCH_SIZE):
            BATCH_X, BATCH_Y = X_train[idx], y_train[idx]

            loss_batch, accuracy_batch, state = model.train(BATCH_X, BATCH_Y, state)
            loss_train += loss_batch
            accuracy_train += accuracy_batch

            print("EPOCH: " + str(epoch) + "BATCH_INDEX:" + str(idx) + "Batch Loss:" + str(
                loss_batch) + "Batch Accuracy:" + str(accuracy_train))

        loss_train /= n_train
        accuracy_train /= n_train

        model.save(epoch)  # save your model after each epoch
        rec_loss.append([loss_train, accuracy_train])


    print("Training Complete")

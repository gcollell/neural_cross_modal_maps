import numpy as np
import tensorflow as tf


class Mapping:
    '''
    This class learns a mapping from training data X to Y
    '''

    def __init__(self, model_type, dataset, par, X_tr, Y_tr):

        self.n_samples = X_tr.shape[0]
        self.par = par
        self.n_batches = int(np.floor(self.n_samples / par['batch_size']))
        self.dataset = dataset

        # ------- DEFINE MODEL -------- #
        tf.reset_default_graph()

        # - PLACEHOLDERS - #
        self.Y_input = tf.placeholder(tf.float32, [None, Y_tr.shape[1]])
        self.X_input = tf.placeholder(tf.float32, [None, X_tr.shape[1]])
        self.pkeep = tf.placeholder(tf.float32) # Probability of keeping a node, for dropout

        if model_type == 'lin':
            self.Y1 = self.X_input
        if model_type == 'nn':
            self.W1 = tf.Variable(self.initializer([X_tr.shape[1], par['n_hidden']]))
            self.b1 = tf.Variable(self.initializer([par['n_hidden']]))
            self.Y1 = self.activation(tf.matmul(self.X_input, self.W1) + self.b1)
            self.Y1 = tf.nn.dropout(self.Y1, self.pkeep) # Dropout

        # - output layer - #
        self.W_out = tf.Variable(self.initializer([int(self.Y1.get_shape()[1]), Y_tr.shape[1]]))
        self.b_out = tf.Variable(self.initializer([Y_tr.shape[1]]))
        self.Y_pred = tf.matmul(self.Y1, self.W_out) + self.b_out

        # - loss & optimizer - #
        self.loss = tf.losses.mean_squared_error(self.Y_input, self.Y_pred)

        self.optimizer = tf.train.AdamOptimizer(learning_rate = par['lr']).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def train_epoch(self, X_tr, Y_tr):

        for i in range(self.n_batches + 1):
            idx = range(i*self.par['batch_size'], min(self.n_samples, (i + 1)*self.par['batch_size']))
            X_batch, Y_batch = X_tr[idx, :], Y_tr[idx, :]
            feed_dict = {self.X_input: X_batch, self.Y_input: Y_batch, self.pkeep: (1-self.par['dropout'])}
            self.sess.run(self.optimizer, feed_dict)


    def predict(self, X):
        Y_pred = self.sess.run(self.Y_pred, {self.X_input: X, self.pkeep: 1})
        return Y_pred


    def initializer(self, shape):
        '''
        Shape is a list [dim_in, dim_out] for matrices, or [dim_out] for biases
        '''
        if len(shape) == 2:
            n_in, n_out = shape[0], shape[1]
            return tf.truncated_normal(shape, stddev= tf.sqrt(2.0/float(n_in + n_out)))
        elif len(shape) == 1:
            if self.par['activation'] in ['sigmoid', 'tanh']:
                return tf.Variable(tf.constant(0.0, shape=shape))
            if self.par['activation'] == 'relu':
                return tf.Variable(tf.constant(0.1, shape=shape))


    def activation(self, input):
        if self.par['activation'] == 'sigmoid':
            return tf.nn.sigmoid(input)
        if self.par['activation'] == 'tanh':
            return tf.nn.tanh(input)
        if self.par['activation'] == 'relu':
            return tf.nn.relu(input)




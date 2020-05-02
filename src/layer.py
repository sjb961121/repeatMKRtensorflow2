import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class MLP(Layer):
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.vars=[]
        self.kernel = self.add_variable('weight', [input_dim, output_dim])
        self.bias = self.add_variable('bias', [output_dim])
        self.vars = [self.kernel,self.bias]

    def call(self, inputs, **kwargs):
        out = tf.matmul(inputs,self.kernel) + self.bias
        out=tf.nn.relu(out)
        return out



class CrossCompressUnit(Layer):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim

        self.weight_vv = self.add_variable(name='weight_vv', shape=(dim, 1), dtype=tf.float32)
        self.weight_ev = self.add_variable(name='weight_ev', shape=(dim, 1), dtype=tf.float32)
        self.weight_ve = self.add_variable(name='weight_ve', shape=(dim, 1), dtype=tf.float32)
        self.weight_ee = self.add_variable(name='weight_ee', shape=(dim, 1), dtype=tf.float32)
        self.bias_v = self.add_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
        self.bias_e = self.add_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]

    def call(self, inputs, **kwargs):
        # [batch_size, dim]
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = tf.expand_dims(v, axis=2)
        e = tf.expand_dims(e, axis=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        # [batch_size, dim]
        v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_transpose, self.weight_ev),
                              [-1, self.dim]) + self.bias_v
        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),
                              [-1, self.dim]) + self.bias_e

        return v_output, e_output

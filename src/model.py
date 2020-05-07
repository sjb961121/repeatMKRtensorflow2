

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from layer import MLP
from layer import CrossCompressUnit



class MKR(Model):
    def __init__(self, args, n_users, n_items, n_entities, n_relations):
        super(MKR, self).__init__()
        self._parse_args(args,n_users, n_items, n_entities, n_relations)
        self._build_inputs()
        self._build_model(args)

    def _parse_args(self, args,n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations
        self.args=args

        # for computing l2 loss
        self.vars_rs = []
        self.vars_kge = []

    def _build_inputs(self):
        self.user_mlp=[]
        self.tail_mlp=[]
        self.cc_unit=[]
        self.kge_mlp=[]

    def _build_model(self,args):
        self._build_low_layers(args)
        self._build_high_layers(args)

    def _build_low_layers(self, args):
        # self.user_emb_matrix = tf.Variable(tf.random.truncated_normal([self.n_user, args.dim]))
        self.user_emb_matrix=tf.keras.layers.Embedding(self.n_user, args.dim)
        # self.item_emb_matrix = tf.Variable(tf.random.truncated_normal([self.n_item, args.dim]))
        self.item_emb_matrix = tf.keras.layers.Embedding(self.n_item, args.dim)
        # self.entity_emb_matrix = tf.Variable(tf.random.truncated_normal([self.n_entity, args.dim]))
        self.entity_emb_matrix = tf.keras.layers.Embedding(self.n_entity, args.dim)
        # self.relation_emb_matrix = tf.Variable(tf.random.truncated_normal([self.n_relation, args.dim]))
        self.relation_emb_matrix = tf.keras.layers.Embedding(self.n_relation, args.dim)
        # self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        # self.item_embeddings = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_indices)
        # self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        # self.relation_embeddings = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        # self.tail_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.tail_indices)

        for _ in range(args.L):
            self.user_mlp.append(MLP(input_dim=args.dim, output_dim=args.dim))
            self.tail_mlp.append(MLP(input_dim=args.dim, output_dim=args.dim))
            self.cc_unit.append(CrossCompressUnit(args.dim))
            # self.user_embeddings = user_mlp(self.user_embeddings)
            # self.item_embeddings, self.head_embeddings = cc_unit([self.item_embeddings, self.head_embeddings])
            # self.tail_embeddings = tail_mlp(self.tail_embeddings)

            # self.vars_rs.extend(self.user_mlp.vars)
            # self.vars_rs.extend(self.cc_unit.vars)
            # self.vars_kge.extend(self.tail_mlp.vars)
            # self.vars_kge.extend(self.cc_unit.vars)

    def _build_high_layers(self, args):
        # RS
            # [batch_size]
        # self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        # self.scores_normalized = tf.nn.sigmoid(self.scores)
        #
        # # KGE
        # # [batch_size, dim * 2]
        # self.head_relation_concat = tf.concat([self.head_embeddings, self.relation_embeddings], axis=1)
        # for _ in range(args.H):
        #     self.kge_mlp = MLP(input_dim=args.dim * 2, output_dim=args.dim * 2)
            # [batch_size, dim]
            # self.head_relation_concat = kge_mlp(self.head_relation_concat)
            # self.vars_kge.extend(kge_mlp.vars)
        for _ in range(self.args.H - 1):
            self.kge_mlp.append(MLP(input_dim=self.args.dim * 2, output_dim=self.args.dim * 2))
        self.kge_pred_mlp = MLP(input_dim=args.dim * 2, output_dim=args.dim)
        # [batch_size, 1]
        # self.tail_pred = kge_pred_mlp(self.head_relation_concat)
        # self.vars_kge.extend(kge_pred_mlp.vars)
        # self.tail_pred = tf.nn.sigmoid(self.tail_pred)
        #
        # self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(self.tail_embeddings * self.tail_pred, axis=1))
        # self.rmse = tf.reduce_mean(
        #     tf.sqrt(tf.reduce_sum(tf.square(self.tail_embeddings - self.tail_pred), axis=1) / args.dim))

    def call(self, inputs, training=None, mask=None):
        pass

    @tf.function
    def train_rs(self, feed_dict):
        self.user_indices=feed_dict[0]
        self.item_indices = feed_dict[1]
        self.labels = tf.cast(feed_dict[2],dtype=tf.float32)
        self.head_indices = feed_dict[3]

        # self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.user_embeddings=self.user_emb_matrix(self.user_indices)
        # self.item_embeddings = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_indices)
        self.item_embeddings = self.item_emb_matrix(self.item_indices)
        # self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        self.head_embeddings = self.entity_emb_matrix(self.head_indices)
        # self.tail_embeddings = tf.zeros(shape=self.head_embeddings)
        for i in range(self.args.L):
            self.user_embeddings = self.user_mlp[i](self.user_embeddings)
            self.item_embeddings, self.head_embeddings = self.cc_unit[i]([self.item_embeddings, self.head_embeddings])
        # self.tail_embeddings=self.tail_mlp(self.tail_embeddings)
        # self.vars_rs = []
        # self.vars_rs.extend(self.user_mlp.vars)
        # self.vars_rs.extend(self.cc_unit.vars)
        # self.vars_kge.extend(self.cc_unit.vars)
        # self.vars_kge.extend(self.tail_mlp.vars)
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.nn.sigmoid(self.scores)

        # self.head_relation_concat = tf.concat([self.head_embeddings, self.relation_embeddings], axis=1)
        # self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
        # self.vars_kge.extend(self.kge_pred_mlp.vars)
        # self.tail_pred = tf.nn.sigmoid(self.tail_pred)

        # self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(self.tail_embeddings * self.tail_pred, axis=1))
        # self.rmse = tf.reduce_mean(
        #     tf.sqrt(tf.reduce_sum(tf.square(self.tail_embeddings - self.tail_pred), axis=1) / self.args.dim))
        # tf.cast(self.labels,tf.float32)
        # print(self.labels)
        # print(self.scores_normalized)

        self.base_loss_rs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        self.l2_loss_rs = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
        for var in self.trainable_variables:
            self.l2_loss_rs += tf.nn.l2_loss(var)
        self.loss_rs = self.base_loss_rs + self.l2_loss_rs * self.args.l2_weight
        # with tf.GradientTape() as tape:
        #     L=self.loss_rs
        #     g=tape.gradient(L,self.vars_rs)
        # optimizers=tf.keras.optimizers.Adam(learning_rate=self.args.lr_rs)
        # optimizers.apply_gradients(grads_and_vars=zip(g, self.vars_rs))
        return self.scores_normalized,self.loss_rs

    @tf.function
    def train_kge(self, feed_dict):
        self.item_indices = feed_dict[0]
        self.head_indices = feed_dict[1]
        self.relation_indices = feed_dict[2]
        self.tail_indices = feed_dict[3]
        # self.item_embeddings = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_indices)
        self.item_embeddings = self.item_emb_matrix(self.item_indices)
        # self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        self.head_embeddings = self.entity_emb_matrix(self.head_indices)
        # self.relation_embeddings = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        self.relation_embeddings = self.relation_emb_matrix(self.relation_indices)
        # self.tail_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.tail_indices)
        self.tail_embeddings = self.entity_emb_matrix(self.tail_indices)

        for i in range(self.args.L):
            self.item_embeddings, self.head_embeddings = self.cc_unit[i]([self.item_embeddings, self.head_embeddings])
            self.tail_embeddings = self.tail_mlp[i](self.tail_embeddings)

        # self.vars_kge = []
        # self.vars_kge.extend(self.cc_unit.vars)
        # self.vars_kge.extend(self.tail_mlp.vars)
        self.head_relation_concat = tf.concat([self.head_embeddings, self.relation_embeddings], axis=1)
        for i in range(self.args.H - 1):
            self.head_relation_concat = self.kge_mlp[i](self.head_relation_concat)
        self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
        # self.vars_kge.extend(self.kge_pred_mlp.vars)
        self.tail_pred = tf.nn.sigmoid(self.tail_pred)

        self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(self.tail_embeddings * self.tail_pred, axis=1))
        self.rmse = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(self.tail_embeddings - self.tail_pred), axis=1) / self.args.dim))
        # with tf.GradientTape() as tape:
        #     L = self.rmse
        #     g = tape.gradient(L, self.vars_kge)
        # optimizers = tf.keras.optimizers.Adam(learning_rate=self.args.lr_kge)
        # optimizers.apply_gradients(zip(g, self.vars_kge))
        self.base_loss_kge = -self.scores_kge
        self.l2_loss_kge = tf.nn.l2_loss(self.head_embeddings) + tf.nn.l2_loss(self.tail_embeddings)
        for var in self.trainable_variables:
            self.l2_loss_kge += tf.nn.l2_loss(var)
        self.loss_kge = self.base_loss_kge + self.l2_loss_kge * self.args.l2_weight
        return self.loss_kge,self.rmse

    def eval(self, feed_dict):
        scores,_ = self.train_rs(feed_dict)
        labels=tf.cast(feed_dict[2],dtype=tf.float32)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc










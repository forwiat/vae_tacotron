import tensorflow as tf
import numpy as np
from modules import get_next_batch, embedding
from hparams import hyperparams
hp = hyperparams()
class Graph:
    def __init__(self, mode='train'):
        self.mode = mode
        self.scope_name = 'vae_tacotron'
        self.reuse = tf.AUTO_REUSE
        if self.mode in ['train', 'eval']:
            self.is_training = True
            self.train()
            tf.summary.scalar('{}/loss'.format(self.mode), self.loss)
            self.merged = tf.summary.merge_all()
            self.t_vars = tf.trainable_variables()
            self.num_paras = 0
            for var in self.t_vars:
                var_shape = var.get_shape().as_list()
                self.num_paras += np.prod(var_shape)
            print('Total number of trainable parameters : %r' % self.num_paras)
        elif self.mode in ['test', 'infer']:
            self.is_training = False
            self.infer()
        else:
            raise Exception('No supported mode in model __init__ function, please check.')
    def train(self):
        self.text, self.refer_mel, self.mel, self.linear = get_next_batch()
        self.encoder_inputs = embedding(self.text, scope='embedding', reuse=self.reuse)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.mel[:, :1, :]), self.mel[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.N_MELS:]
        with tf.variable_scope(self.scope_name):
            self.text_outputs = encoder(self.encoder_inputs, is_training=self.is_training)
            self.vae_outputs, self.mu, self.log_var = vae(self.refer_mel, is_training=self.is_training)
            self.encoder_outputs = self.text_outputs + self.vae_outputs
            self.mel_hat, self.alignments = decoder(self.decoder_inputs,
                                                   self.encoder_outputs,
                                                   is_training=self.is_training)
            self.linear_hat = postnet(self.mel_hat, is_training=self.is_training)
        if self.mode in ['train', 'eval']:
            self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
            self.lr = tf.train.exponential_decay(learning_rate=hp.LR, global_step=self.global_step,
                                                 decay_steps=hp.DECAY_STEPS,
                                                 decay_rate=hp.DECAY_RATE)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_hat - self.mel))
            self.linear_loss = tf.reduce_mean(tf.abs(self.linear_hat - self.linear))
            self.kl_loss = - 0.5 * tf.reduce_sum(1 + self.log_var - tf.pow(self.mu, 2) - tf.exp(self.log_var))
            self.vae_loss_weight = control_weight(self.global_step)
            self.loss = self.mel_loss + self.linear_loss + self.vae_loss_weight * self.kl_loss
            self.


    def eval(self):
        pass
    def infer(self):
        pass

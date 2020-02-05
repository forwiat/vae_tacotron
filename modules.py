import tensorflow as tf
from hparams import hyperparams
import glob
hp = hyperparams()
def get_next_batch():
    tfrecords = glob.glob(f'{hp.TRAIN_DATASET_PATH}/*.tfrecord')
    filename_queue = tf.train.string_input_producer(tfrecords, shuffle=True, num_epochs=hp.NUM_EPOCHS)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'text': tf.FixedLenFeature([], tf.int64),
            'refer_mel': tf.FixedLenFeature([], tf.float32),
            'mel': tf.FixedLenFeature([], tf.float32),
            'linear': tf.FixedLenFeature([], tf.float32)
        }
    )
    text = tf.reshape(features['text'], [])
    refer_mel = tf.reshape(features['refer_mel'], [])
    mel = tf.reshape(features['mel'], [])
    linear = tf.reshape(features['linear'], [])
    text_batch, refer_mel_batch, mel_batch, linear_batch = tf.train.batch([text, refer_mel, mel, linear])
    return text_batch, refer_mel_batch, mel_batch, linear_batch

def embedding(inputs, zero_pad=True, scope='embedding', reuse=None):
    '''
    :param inputs: A 2-d tensor. [N, L]. text ids.
    :param zero_pad: Boolean. Embed matrix first zero line.
    :param scope: String. Scope name.
    :param reuse: Boolean. Default tf.AUTO_REUSE.
    :return: A 3-d tensor. [N, L, EMBED_SIZE].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[hp.VOCAB_SIZE, hp.EMBED_SIZE],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, hp.EMBED_SIZE]), lookup_table[1:, :]), 0)
        return tf.nn.embedding_lookup(lookup_table, inputs)


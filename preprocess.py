import librosa
import multiprocessing as mp
import tensorflow as tf
import os
from hparams import hyperparams
from tqdm import tqdm
from utils import get_spectrograms, match_vocab
hp = hyperparams()

def process(args):
    (tfid, split_dataset) = args
    writer = tf.python_io.TFRecordWriter(os.path.join(hp.TRAIN_DATASET_PATH, f'train_{tfid}.tfrecord'))
    for i in tqdm(split_dataset):
        text = i[0]
        fpath = i[1]
        idxs = match_vocab(text)
        mel, mag = get_spectrograms(fpath)
        example = tf.train.Example(features=tf.train.Features(feature={
            'x': tf.train.Feature(int64_list=tf.train.Int64List(value=idxs.reshape(-1))),
            'y': tf.train.Feature(float_list=tf.train.FloatList(value=mel.reshape(-1))),
            'z': tf.train.Feature(float_list=tf.train.FloatList(value=mag.reshape(-1)))
        }))


def main():


if __name__ == '__main__':
    main()

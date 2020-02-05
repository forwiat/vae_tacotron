import librosa
import numpy as np
from hparams import hyperparams
hp = hyperparams()

def get_spectrograms(fpath: str):
    y, _ = librosa.load(fpath, sr=hp.SR)
    y, _ = librosa.effects.trim(y)
    y = np.append(y[0], y[1:] - hp.PREEMPHASIS * y[:-1])
    linear = librosa.stft(y=y, n_fft=hp.N_FFT, win_length=hp.WIN_LENGTH, hop_length=hp.HOP_LENGTH)
    mag = np.abs(linear) # [1+n_fft//2, T]
    mel_basis = librosa.filters.mel(hp.SR, hp.N_FFT, hp.N_MELS) # [n_mels, 1+n_fft//2]
    mel = np.dot(mel_basis, mag) # [n_mels, T]
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mel = np.clip((mel - hp.REF_DB + hp.MAX_DB) / hp.MAX_DB, 1e-8, 1)
    mag = np.clip((mag - hp.REF_DB + hp.MAX_DB) / hp.MAX_DB, 1e-8, 1)
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)
    return mel, mag

def match_vocab(text: str):
    char2idx = {char: idx for char, idx in enumerate(hp.VOCAB)}
    for i in range(len(text)):
        if text[i] not in char2idx.keys():
            print(f'Exit {text[i]} not in hp.VOCAB, please check.')
            exit(0)
    idxs = [char2idx[text[i]] for i in range(len(text))]
    return idxs


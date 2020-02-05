class hyperparams:
      def __init__(self):
          self.CSV_PATH = ''
          self.MULTI_PROCESS = False
          self.PREEMPHASIS = 0.97
          self.N_FFT = 1024
          self.SR = 16000
          self.WIN_LENGTH = int(self.SR * 0.05)
          self.HOP_LENGTH = int(self.SR * 0.0125)
          self.N_MELS = 80
          self.REF_DB = 20
          self.MAX_DB = 100
          self.VOCAB = 'abcdefghijklmnopqrstuvwxyz12345,.?!'
          self.TRAIN_DATASET_PATH = './train_data'
          self.EVAL_DATASET_PATH = './eval_data'
          self.TRAIN_DATASET_RATE = 0.9
          self.NUM_EPOCHS = 30
          self.EMBED_SIZE = 256
          self.VOCAB_SIZE = self.GET_VOCAB_SIZE()
          self.LR = 0.001
          self.DECAY_RATE = 0.5
          self.DECAY_STEPS = 1000
      def GET_VOCAB_SIZE(self):
          return len(self.VOCAB)

import os
from model_settings import ModelSettings

class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        self.TRAINING_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'training-lmdb')
        self.VALIDATION_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'test-lmdb')

        self.TRAIN_CNN = True

        self.OPTIMIZER = 'Adam'   # one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.LEARNING_RATE = 0.001
        self.LR_DROP_FACTOR = 0
        self.LR_DROP_PATIENCE = 20
        self.WEIGHT_DECAY = 0      # use 0 to disable it
        self.CLIP_GRAD_NORM = 0      # max l2 norm of gradient of parameters - use 0 to disable it
        self.HORIZONTAL_FLIPPING = True
        self.RANDOM_JITTER = True
        self.CRITERION = 'Dice'       # One of 'CE', 'Dice', 'Multi'
        self.OPTIMIZE_BG = False

        self.SEED = 13

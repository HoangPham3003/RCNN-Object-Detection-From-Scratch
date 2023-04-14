import torch

class Config:
    def __init__(self):
        # Config
        self.NUM_CLASSES = 5
        self.CLASSES = ['nothing', 'pineapple', 'snake fruit', 'dragon fruit', 'banana']
        self.COLORS = ['black', 'red', 'green', 'blue', 'yellow']
            
        self.IMG_FOLDER = "./images"
        self.ANNOT_FOLDER = "./annotations"

        self.MAX_PROPOSED_ROI = 2000

        self.LIMIT_POS_PROP_ROI = 40
        self.LIMIT_NEG_PROP_ROI = 10

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IMG_SIZE = 384
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.0001

        self.MAX_PROP_INFER_ROI = 1000

        self.NMS_THRESHOLD = 0.1
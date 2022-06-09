import torch
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda")

class PLADGen:

    def __init__(
        self,
        path,
        batch_size,
        train_size,
        mode,
        num_inst_progs
    ):

        self.num_inst_progs = num_inst_progs
        
        self.infer_labels = torch.from_numpy(torch.load(path + "infer_labels.pt")).long()
        self.infer_images = torch.from_numpy(torch.load(path + "infer_images.pt")).float()
        self.target_images = torch.from_numpy(torch.load(path + "target_images.pt")).float()
        
        # pad labels with a stop symbol
        self.infer_labels = F.pad(self.infer_labels, (0, 1), 'constant', 399)

        self.train_size = train_size
        self.batch_size = batch_size
        self.mode = mode

        self.rng = torch.arange(self.batch_size).long()
        
    def get_train_data(self):
        while True:
            ids = np.arange(self.train_size)
            np.random.shuffle(ids)
            for i in range(0, self.train_size, self.batch_size):
                
                inst_idxs = torch.randint(0, self.num_inst_progs, (self.batch_size,)).long()
                
                exp_batch_labels = self.infer_labels[ids[i:i+self.batch_size]]
                exp_batch_infer_images = self.infer_images[ids[i:i+self.batch_size]]
                
                batch_labels = exp_batch_labels[self.rng, inst_idxs]
                batch_infer_images = exp_batch_infer_images[self.rng, inst_idxs]
                
                batch_target_images = self.target_images[ids[i:i+self.batch_size]]
                
                if self.mode == 'LEST':
                    batch_data = batch_infer_images

                elif self.mode == 'ST':
                    batch_data = batch_target_images

                else:
                    assert False, f'{self.mode}'
                    
                yield (batch_data, batch_labels)


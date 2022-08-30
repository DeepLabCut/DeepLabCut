"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
from torch.utils.data import Dataset
import numpy as np


class TripletDataset(Dataset):
    def __init__(self, datasource, transform=None):

        self.x = datasource

        # normalize vectors here
        """
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                # i th vector at j th kpt
                v = self.x[i][j]
                normalized_v = v / np.sqrt(np.sum(v**2))
                self.x[i][j] = normalized_v
        """

        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        anchor, pos, neg = self.x[index]

        anchor = anchor.astype(np.float32)
        pos = pos.astype(np.float32)
        neg = neg.astype(np.float32)

        if self.transform is not None:
            # maybe needs to convert them to embeddings and position token

            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg

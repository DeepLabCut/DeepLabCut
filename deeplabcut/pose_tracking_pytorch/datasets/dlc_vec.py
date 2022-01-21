from torch.utils.data import Dataset
import numpy as np
import pickle


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


class PairDataset(Dataset):
    def __init__(self, datasource, transform=None):
        self.datasource = datasource
        with open(datasource, "rb") as f:
            data = pickle.load(f)
            self.x = data["vectors"]
            self.y = data["gts"]

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        vec1, vec2 = self.x[index]
        gt1, gt2 = self.y[index]
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)

        if self.transform is not None:
            # maybe needs to convert them to embeddings and position token

            vec1 = self.transform(vec1)
            vec2 = self.transform(vec2)

        return (vec1, gt1), (vec2, gt2)

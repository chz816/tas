import torch
from torch.utils.data import Dataset
import scipy.sparse

class CTMDataset(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, X, idx2token):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """

        self.X = X
        self.idx2token = idx2token

    def __len__(self):
        """Return length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X[i]) == scipy.sparse.csr.csr_matrix:
            X = torch.FloatTensor(self.X[i].todense())
        else:
            X = torch.FloatTensor(self.X[i])

        return {'X': X}



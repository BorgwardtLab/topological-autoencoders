from sklearn.datasets import make_s_curve, make_swiss_roll
from torch.utils.data import Dataset

class SwissRoll(Dataset):
    def __init__(self, n_samples=6000, noise=0.1, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.data = make_swiss_roll(n_samples, noise, seed)[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SCurve(Dataset):
    def __init__(self, n_samples=6000, noise=0.1, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.data = make_s_curve(n_samples, noise, seed)[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

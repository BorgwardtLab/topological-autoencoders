"""Datasets."""
import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

import requests
import zipfile 
import io
import glob
import pandas as pd
import numpy as np
import matplotlib.image as mpimg


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

class COIL100Base(Dataset):
    """Rotated Objects Dataset. Base Class for downloading and loading the data"""

    def __init__(self, root, transform=None, train=True, download=True):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

        if not os.path.exists(self.root + '/coil-100'):
            self.download()

        self.data, self.labels = self.load_data()            
    
    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        #TODO: check if images are in correct dimension ordering!
        if self.transform is not None:
            img = self.transform(img)        
        return img, target

    def download(self):
        url = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/'
        name = 'coil-100.zip' 
        file_path = path+name
        results = requests.get(file_path)
        z = zipfile.ZipFile(io.BytesIO(results.content))
        z.extractall(self.root)

    def load_data(self): 
        filelist = glob.glob(self.root + '/coil-100/*.png')
        #labels are sorted according to filelist:
        labels = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
        labels = [int(i) for i in labels.values]
        data = []
        for filename in filelist:
            im = mpimg.imread(filename)
            print(f'IM SHAPE: {im.shape}')
            #imt = np.transpose(im, (2, 0, 1)) # for consistency with other datasets put color channels first
            #print(f'IMT SHAPE: {imt.shape}')
            #data.append(imt) 
            data.append(im) #imshow doesnt take it like that, transpose back for that
        return np.stack(data), labels


class COIL100(COIL100Base):
    """Rotated Objects Dataset."""

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """COIL100 dataset normalized."""

        super().__init__(
            BASEPATH, transform=self.transforms, train=train, download=True)



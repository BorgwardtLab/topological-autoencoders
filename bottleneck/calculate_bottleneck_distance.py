#!/usr/bin/env python3


import glob
import os

import numpy as np
import pandas as pd

batch_size = 128


if __name__ == '__main__':

    for root, dirs, files in os.walk('./'):
        data_sets = dirs

        for data_set in data_sets:
            original_data = pd.read_csv(
                os.path.join(root, data_set, 'data.csv'),
                header=None
            )

            original_data = original_data.values

            files = sorted(glob.glob(
                    os.path.join(root, data_set, '*_latents.csv')
                )
            )

            random_indices = np.random.choice(
                original_data.shape[0],
                batch_size,
                replace=False
            )

            for filename in files:
                name = os.path.basename(filename)
                name = name[:name.find('_')]

                latent_space = pd.read_csv(
                    filename,
                    header=0
                )

                latent_space = latent_space[['0', '1']]
                latent_space = latent_space.values

                print(latent_space)

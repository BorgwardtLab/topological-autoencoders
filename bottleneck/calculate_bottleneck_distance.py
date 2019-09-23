#!/usr/bin/env python3


import glob
import os
import subprocess

import numpy as np
import pandas as pd

batch_size = 64


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

            X_sample = original_data[random_indices]
            np.savetxt('/tmp/X.csv', X_sample, delimiter=' ')

            diagram = subprocess.run(
                ['vietoris_rips',
                '-n',
                '/tmp/X.csv',
                '1e8',
                '1'],
                capture_output=True,
            )

            diagram = diagram.stdout
            diagram = diagram.decode('utf-8')

            with open('/tmp/D1.txt', 'w') as f:
                f.write(diagram)

            for filename in files:
                name = os.path.basename(filename)
                name = name[:name.find('_')]

                latent_space = pd.read_csv(
                    filename,
                    header=0
                )

                latent_space = latent_space[['0', '1']]
                latent_space = latent_space.values

                Y_sample = latent_space[random_indices]

                np.savetxt('/tmp/Y.csv', Y_sample, delimiter=' ')

                diagram = subprocess.run(
                    ['vietoris_rips',
                    '-n',
                    '/tmp/Y.csv',
                    '1e8',
                    '1'],
                    capture_output=True,
                )

                diagram = diagram.stdout
                diagram = diagram.decode('utf-8')

                with open('/tmp/D2.txt', 'w') as f:
                    f.write(diagram)
                
                bottleneck = subprocess.run(
                    ['topological_distance',
                    '-b',
                    '/tmp/D1.txt',
                    '/tmp/D2.txt'
                    ],
                    capture_output=True,
                )

                bottleneck = bottleneck.stdout
                bottleneck = bottleneck.decode('utf-8')

                bottleneck = bottleneck.split('\n')[0]
                bottleneck = bottleneck.split(' ')
                bottleneck = float(bottleneck[1])

                print(data_set, name, bottleneck)


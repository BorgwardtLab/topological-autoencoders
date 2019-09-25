#!/usr/bin/env python3


import collections
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

batch_size = int(sys.argv[1])
n_iterations = 3


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

            bottlenecks = collections.defaultdict(list)

            for i in range(n_iterations):

                # Ensures that we never take more than the number of
                # samples, regardless of the batch size parameter. 
                if original_data.shape[0] < batch_size:
                    batch_size = original_data.shape[0]

                random_indices = np.random.choice(
                    original_data.shape[0],
                    batch_size,
                    replace=False
                )

                X_sample = original_data[random_indices]
                np.savetxt('/tmp/Xw.csv', X_sample, delimiter=' ')

                diagram = subprocess.run(
                    ['vietoris_rips',
                    '-n',
                    '/tmp/Xw.csv',
                    '1e8',
                    '1'],
                    capture_output=True,
                )

                diagram = diagram.stdout
                diagram = diagram.decode('utf-8')

                with open('/tmp/D1w.txt', 'w') as f:
                    f.write(diagram)

                D1 = np.genfromtxt('/tmp/D1w.txt')

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

                    np.savetxt('/tmp/Yw.csv', Y_sample, delimiter=' ')

                    diagram = subprocess.run(
                        ['vietoris_rips',
                        '-n',
                        '/tmp/Yw.csv',
                        '1e8',
                        '1'],
                        capture_output=True,
                    )

                    diagram = diagram.stdout
                    diagram = diagram.decode('utf-8')

                    with open('/tmp/D2w.txt', 'w') as f:
                        f.write(diagram)

                    D2 = np.genfromtxt('/tmp/D2w.txt')

                    bottleneck = subprocess.run(
                        ['topological_distance',
                        '-w',
                        '-p',
                        '1',
                        '/tmp/D1w.txt',
                        '/tmp/D2w.txt'
                        ],
                        capture_output=True,
                    )

                    bottleneck = bottleneck.stdout
                    bottleneck = bottleneck.decode('utf-8')

                    bottleneck = bottleneck.split('\n')[0]
                    bottleneck = bottleneck.split(' ')
                    bottleneck = float(bottleneck[1])

                    bottlenecks[name].append(bottleneck)

                    #l2 = np.linalg.norm(D1 - D2)
                    #print(data_set, name, l2)

            for name in sorted(bottlenecks.keys()):
                print(batch_size,
                      data_set,
                      name,
                      np.mean(bottlenecks[name]),
                      np.std(bottlenecks[name])
                )
                sys.stdout.flush()
            print('')


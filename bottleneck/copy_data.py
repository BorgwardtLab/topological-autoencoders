import glob
import os
import sys

from shutil import copyfile


if __name__ == '__main__':
    root = sys.argv[1]

    for root, dirs, files in os.walk(root):
        data_sets = dirs

        for data_set in data_sets:
            data_set_path = os.path.join(root, data_set)

            methods = sorted(glob.glob(os.path.join(data_set_path,
                '*')))

            for method in methods:
                name = os.path.basename(method)

                print(data_set)

                copyfile(
                    os.path.join(method, 'latents.csv'),
                    os.path.join('.', data_set, name + '_latents.csv')
                )

        break

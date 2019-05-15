"""Create config files for dataset model combinations."""
import argparse
import itertools
import subprocess
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', required=True, nargs='+', type=str)
    parser.add_argument('--models', required=True, nargs='+', type=str)
    parser.add_argument('--output', required=True, type=str)

    args = parser.parse_args()

    for dataset, model in itertools.product(args.datasets, args.models):
        output_path = os.path.join(args.output, dataset)
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, f'{model}.json')
        subprocess.call([
            'python', '-m', 'exp.hyperparameter_search', 'save_config', 'with',
            dataset, model, f'config_filename={output_filename}'
        ])


if __name__ == '__main__':
    main()

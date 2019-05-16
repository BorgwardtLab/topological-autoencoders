"""Create config files for dataset model combinations."""
import argparse
import itertools
import subprocess
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sacred_experiment')
    parser.add_argument('--name', type=str, required=True, action='append')
    parser.add_argument('--set', required=True, nargs='+', type=str,
                        action='append')
    parser.add_argument('--output-pattern', required=True, type=str)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    assert len(args.name) == len(args.set)

    for possible_parameters in itertools.product(*args.set):
        parameter_mapping = dict(zip(args.name, possible_parameters))
        output_filename = args.output_pattern.format(**parameter_mapping)
        if not args.overwrite and os.path.exists(output_filename):
            print(f'Skipping... {output_filename} already exists.')
            continue
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        subprocess.call(
            [
                'python', '-m', args.sacred_experiment, 'save_config', 'with',
                f'config_filename={output_filename}'
            ] + list(possible_parameters)
        )


if __name__ == '__main__':
    main()

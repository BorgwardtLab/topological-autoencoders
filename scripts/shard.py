"""Simple python script to do sharding for multiple gpus."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_values', nargs='+', type=str)
    parser.add_argument('--nshard', type=int, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    print(' '.join(args.list_values[args.index::args.nshard]))


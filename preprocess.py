import pandas as pd
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str, help="Relative path for the dataset.")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

if __name__ == "__main__":
    main()

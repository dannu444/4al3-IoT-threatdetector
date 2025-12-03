import pandas as pd
from argparse import ArgumentParser
import random

def preprocess(df:pd.DataFrame, downsample:bool=False, sample:bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Get number of classes

    num_classes = len(df["Attack_type"].unique())

    # If requested, downsample the data

    if downsample:
        classes = df["Attack_type"]
        value_counts = classes.value_counts()
        min_class_num = value_counts.min()

        selected_indices_overall = []
        for c in classes.unique():
            indices = df.query(f"Attack_type == '{c}'").index.tolist()
            selected_indices = random.sample(indices, min_class_num)
            selected_indices_overall.extend(selected_indices)

        df = df.loc[selected_indices_overall]

    if sample:
        indices = df.query(f"Attack_type == 'DOS_SYN_Hping'").index.tolist()
        selected_indices = random.sample(indices, 10000)
        remove_indices = []
        for i in indices:
            if i not in selected_indices:
                remove_indices.append(i)
        df.drop(remove_indices, axis=0, inplace=True)

    # Remove id related features and bwd_URG_flag_count (constant feature)

    df.drop(["id", "id.orig_p", "id.resp_p", "bwd_URG_flag_count"], axis=1, inplace=True)

    # Encode categorical features to numerical values

    df = pd.get_dummies(df, dtype=int)

    # Split data into input and target

    input = df.iloc[:, :df.shape[1] - num_classes]
    target = df.iloc[:, df.shape[1] - num_classes:]

    # Perform z-score normalization on input data

    for col in input.columns:
        input[col] = (input[col] - input[col].mean()) / input[col].std()

    # Return input and target DataFrames

    return input, target

def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str, help="Relative path for the dataset (csv file).")
    parser.add_argument("input_dest_path", type=str, help="Relative path to save preprocessed input (csv file).")
    parser.add_argument("target_dest_path", type=str, help="Relative path to save preprocessed target (csv file).")
    parser.add_argument("-d", "--down_sample", action="store_true", help="Flag for whether to perform downsampling.")
    parser.add_argument("-s", "--sample", action="store_true", help="Randomly samples 10,000 instances of the dos_syn_hping class to reduce bias.")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    input, target = preprocess(df, args.down_sample, args.sample)
    input.to_csv(args.input_dest_path, index=False)
    target.to_csv(args.target_dest_path, index=False)

if __name__ == "__main__":
    main()

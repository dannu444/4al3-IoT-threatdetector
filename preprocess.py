import pandas as pd
from argparse import ArgumentParser

def preprocess(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Get number of classes

    num_classes = len(df["Attack_type"].unique())

    # Remove id related features

    df.drop(["id", "id.orig_p", "id.resp_p"], axis=1, inplace=True)

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
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    input, target = preprocess(df)
    input.to_csv(args.input_dest_path, index=False)
    target.to_csv(args.target_dest_path, index=False)

if __name__ == "__main__":
    main()

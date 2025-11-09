from argparse import ArgumentParser
import pandas as pd

def check_correlation(df:pd.DataFrame, threshold:float) -> None:
    # Get correlations of features
    correlations_df = df.corr()
    correlations_names = correlations_df.columns.to_list()

    # Get feature pairs with correlation greater than or equal to the threshold
    high_corr = []
    correlations = correlations_df.to_numpy()
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[1]):
            if correlations[i, j] >= threshold:
                    high_corr.append((i, j))

    # Print out correlated features according to threshold
    for i, j in high_corr:
        print(f"{correlations_names[i]} has a correlation with {correlations_names[j]} above {threshold}")

def main():
    parser = ArgumentParser()
    parser.add_argument("file_path", type=str, help="Relative path for the dataset.")
    parser.add_argument("correlation_threshold", type=float, help="Checks for correlation equal to or greater than this value.")
    args = parser.parse_args()

    # Get dataset
    df = pd.read_csv(args.file_path)

    # Check correlation
    check_correlation(df, args.correlation_threshold)

    # Print DataFrame Info
    print("---------- DataFrame Info ----------")
    df.info()

if __name__ == "__main__":
    main()

from argparse import ArgumentParser
import pandas as pd
from ydata_profiling import ProfileReport

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
    parser.add_argument("-c", "--correlation_threshold", type=float, help="Checks for correlation equal to or greater than this value. Note that to use this feature the dataset must only have numerical data.")
    parser.add_argument("-p", "--profile_ydata", action="store_true", help="Flag for whether to perform dataset profiling with ydata_profiling.")
    args = parser.parse_args()

    # Get dataset
    df = pd.read_csv(args.file_path)

    if args.profile_ydata:
        # ydata profiling
        profile = ProfileReport(df, minimal=True)
        profile.to_file(f"{args.file_path.removesuffix(".csv")}.html")

    if args.correlation_threshold:
        # Check correlation
        print("---------- Correlation Info ----------")
        check_correlation(df, args.correlation_threshold)

    # Print DataFrame Info
    print("---------- DataFrame Info ----------")
    df.info()

if __name__ == "__main__":
    main()

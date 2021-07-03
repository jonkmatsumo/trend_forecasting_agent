import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse


def main():
    # get arguments from user
    args = parse_args()

    df = read_csv(args.input_csv)
    print(args.input_csv + ":")
    print(df.head())

    included_rows = read_file(args.input_features)
    df_filtered = df[included_rows]
    print(args.output_csv + ":")
    print(df_filtered.head())
    df_filtered.to_csv(args.output_csv)

    output_csv_norm = args.output_csv.split('.')[0] + '_normalized.csv'

    scaler = MinMaxScaler()
    for i in range(1, len(included_rows)):
        keyword = included_rows[i]
        df_filtered[keyword] = scaler.fit_transform(df_filtered[keyword].values.reshape(-1, 1))
    print(output_csv_norm + ":")
    print(df_filtered.head())
    df_filtered.to_csv(output_csv_norm)
    return 0


def read_csv(filename):
    df = pd.read_csv(filename)
    print(df.head())
    return df


def read_file(filename):
    file = open(filename, 'r')
    lines = file.read().splitlines()
    file.close()
    return lines


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_csv", help="path to read dataset from")
    arg_parser.add_argument("input_features", help="path to read features from")
    arg_parser.add_argument("output_csv", help="path to write trend results to")
    return arg_parser.parse_args()


if __name__ == '__main__':
    main()
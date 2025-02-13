import pandas as pd
import argparse
import os
import re


def load_dataframe_from_directory(directory_path):
    pattern = re.compile(r'human_metadata_\d+\.pkl')
    dataframes = []

    for filename in os.listdir(directory_path):
        if pattern.match(filename):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_pickle(file_path)
            dataframes.append(df)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return RuntimeError("Column not find dataframes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory_path', type=str, help='Path to the directory')
    parser.add_argument('columns', type=str, nargs='+', help='List of columns')
    args = parser.parse_args()

    print(f"Directory path provided: {args.directory_path}")

    import json

    df = load_dataframe_from_directory(args.directory_path)
    with open("unique_condtions.json", "w") as f:
        json.dump({c: pd.unique(df[c]) for c in args.columns }, f)

#!/usr/bin/env python3
"""
Process Goodreads interactions CSV:
1. Load dataset with Dask for large-scale processing.
2. Filter only read books with ratings.
3. Group by users and count unique books read.
4. Identify users below the 95th percentile of reading activity.
"""

import argparse
import os
import sys
import dask.dataframe as dd


def load_data(file_path):
    """
    Load CSV data into a Dask DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dd.DataFrame: Loaded Dask DataFrame.
    """
    return dd.read_csv(file_path)


def clean_data(df):
    """
    Filter the DataFrame to include only read books with ratings.

    Args:
        df (dd.DataFrame): Raw interactions DataFrame.

    Returns:
        dd.DataFrame: Filtered DataFrame with user_id, book_id, and rating.
    """
    return df.loc[df['is_read'] == 1, ['user_id', 'book_id', 'rating']]


def compute_books_per_user(df):
    """
    Count unique books read per user.

    Args:
        df (dd.DataFrame): Filtered DataFrame.

    Returns:
        dd.Series: Number of unique books per user.
    """
    return df.groupby('user_id')['book_id'].nunique()


def filter_valid_users(books_per_user, percentile=0.95):
    """
    Filter users at or below a given percentile of reading activity.

    Args:
        books_per_user (dd.Series): Series of books read per user.
        percentile (float): Percentile threshold.

    Returns:
        list: List of valid user IDs.
    """
    threshold = books_per_user.quantile(percentile).compute()
    return books_per_user[books_per_user <= threshold].index.compute().tolist()


def parse_arguments():
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Read raw data and transform to data frames")
    parser.add_argument(
        "file_path",
        help="Path to the raw data file."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.95,
        help="Percentile threshold for filtering users (default: 0.95)"
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Validate input file
    if not os.path.exists(args.file_path):
        sys.exit(f"❌ Error: File not found - {args.file_path}")

    # Load and process data
    df = load_data(args.file_path)
    df_clean = clean_data(df)
    books_per_user = compute_books_per_user(df_clean)
    valid_users = filter_valid_users(books_per_user, args.percentile)

    print(f"✅ Found {len(valid_users)} valid users at or below the {args.percentile*100:.0f}th percentile.")
    # Optionally, you can save valid_users to a file or process further


if __name__ == "__main__":
    main()

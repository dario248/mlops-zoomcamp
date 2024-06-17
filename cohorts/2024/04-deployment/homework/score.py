#!/usr/bin/env python
# coding: utf-8
import pickle
import argparse
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def load_model(model_path: str) -> Tuple[DictVectorizer, LinearRegression]:
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    return df

def apply_model(df: pd.DataFrame,
                dv: DictVectorizer,
                model: LinearRegression) -> np.ndarray:
    # Pre-process data
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    # Get predictions 
    y_pred = model.predict(X_val)

    return y_pred

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Batch scoring script",
    )
    parser.add_argument('--year', type=int, required=True,
                        help='Dataset year')
    parser.add_argument('--month', type=int, required=True, 
                        help='Dataset month')
    args = parser.parse_args()

    return args

def run():
    args = get_args()

    input_file = f'./input/yellow/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    output_file = f"output/yellow/{args.year:04d}-{args.month:02d}-predictions.parquet"
    input_df = read_data(input_file)
    input_df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + input_df.index.astype('str')

    dv, model = load_model('model.bin')

    y_pred = apply_model(input_df, dv, model)

    print(f"Mean predicted duration: {y_pred.mean():.3f}")
    # Store results
    df_result = pd.DataFrame()
    df_result["ride_id"] = input_df["ride_id"]
    df_result["predictions"] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == "__main__":
    run()
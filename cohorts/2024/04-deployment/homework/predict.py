import pickle
import pandas as pd
from flask import Flask, request, jsonify
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def load_model(model_path: str) -> Tuple[DictVectorizer, LinearRegression]:
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model

def read_data(input_data: dict,
              categorical: list[str]) -> pd.DataFrame:
    year = input_data["year"]
    month = input_data["month"]

    input_file = f"./input/yellow/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    # Load input data
    df = pd.read_parquet(input_file)
    # Pre-process input data
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df

def predict(input_dicts: list[dict], 
            model: LinearRegression,
            dv: DictVectorizer):
    X_val = dv.transform(input_dicts)
    preds = model.predict(X_val)

    return float(preds.mean())

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    input_data = request.get_json()
    categorical = ['PULocationID', 'DOLocationID']
    input_df = read_data(input_data, categorical)
    
    dv, model = load_model(input_data["model"])
    features = input_df[categorical].to_dict(orient='records')
    pred = predict(features, model, dv)

    result = {
        'mean duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
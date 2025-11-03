import pandas as pd
import numpy as np
import pickle

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c

def preprocess(df: pd.DataFrame, freq_maps_path=None):
    df = df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['dayofweek'] = df['transaction_time'].dt.dayofweek
    df['month'] = df['transaction_time'].dt.month
    df['distance'] = haversine_km(df['lat'], df['lon'], df['merchant_lat'], df['merchant_lon'])
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)

    if freq_maps_path:
        with open(freq_maps_path, "rb") as f:
            freq_maps = pickle.load(f)
        for col, mapping in freq_maps.items():
            df[col + '_freq'] = df[col].map(mapping).fillna(0)

    drop_cols = ['transaction_time','name_1','name_2',
                 'merch','cat_id','one_city','us_state',
                 'lat','lon','merchant_lat','merchant_lon',
                 'street','jobs']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

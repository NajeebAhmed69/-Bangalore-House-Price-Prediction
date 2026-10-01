import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load Data
df = pd.read_csv('Bengaluru_House_Data.csv')

# 2. Extract BHK and Clean Total Square Feet
df['bhk'] = df['size'].apply(
    lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else None
)


def convert_sqft_to_num(val):
  tokens = str(val).split('-')
  if len(tokens) == 2:
    return (float(tokens[0]) + float(tokens[1])) / 2
  try:
    return float(val)
  except:
    return None


df['sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# 3. Clean Missing Values & Standardize Locations
df = df.dropna(subset=['location', 'sqft', 'bath', 'bhk']).copy()
df['location'] = df['location'].apply(lambda x: x.strip())

# Aggregate rare locations (<= 10 listings) into 'other'
loc_counts = df['location'].value_counts()
df['location'] = df['location'].apply(
    lambda x: 'other' if loc_counts.get(x, 0) <= 10 else x
)

# 4. Outlier Removal (Domain-Specific)
# Rule A: Minimum 300 sqft per bedroom
df = df[~(df['sqft'] / df['bhk'] < 300)]

# Rule B: Remove price-per-sqft anomalies beyond 1 standard deviation per locality
df['price_per_sqft'] = df['price'] * 100000 / df['sqft']


def remove_pps_outliers(data):
  df_out = pd.DataFrame()
  for _, subdf in data.groupby('location'):
    m = np.mean(subdf.price_per_sqft)
    st = np.std(subdf.price_per_sqft)
    reduced = subdf[
        (subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))
    ]
    df_out = pd.concat([df_out, reduced], ignore_index=True)
  return df_out


df = remove_pps_outliers(df)

# Rule C: Bathrooms should not exceed bedrooms + 2
df = df[df['bath'] < df['bhk'] + 2]

# 5. Define Feature Matrix & Target
features = ['location', 'sqft', 'bath', 'bhk']
X = df[features]
y = df['price']  # Price is quoted in Lakhs (INR)

# 6. Build Scikit-Learn Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            'loc_encoder',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            ['location'],
        )
    ],
    remainder='passthrough',
)

pipeline = Pipeline(
    steps=[
    ('preprocessor', preprocessor), 
    ('regressor', LinearRegression())
    ]
)

# 7. Train & Evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f'R² Score : {r2_score(y_test, y_pred):.4f}')
print(f'MAE      : ₹{mean_absolute_error(y_test, y_pred):,.2f} Lakhs')

# 8. Export Model Pipeline & Unique Locations List
pickle.dump(pipeline, open('bangalore_house_model.pkl', 'wb'))

locations_list = sorted([loc for loc in X['location'].unique() if loc != 'other'])
pickle.dump(locations_list, open('locations.pkl', 'wb'))
print('Model and location artifacts saved successfully!')
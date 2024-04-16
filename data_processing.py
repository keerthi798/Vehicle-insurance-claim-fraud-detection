import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df_train_full = pd.read_csv("E:/S2_network/project4/pythonProject3/data/raw/train.csv")
df_test = pd.read_csv("E:/S2_network/project4/pythonProject3/data/raw/test.csv")

df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=99)

df_train = df_train[df_train["fraud"] != -1]
df_val = df_val[df_val["fraud"] != -1]

for df in [df_train, df_val, df_test]:
    df.loc[df["age_of_driver"] > 100, "age_of_driver"] = np.nan
    df.loc[df["annual_income"] == -1, "annual_income"] = np.nan
    df.loc[df["zip_code"] == 0, "zip_code"] = np.nan

for df in [df_train, df_val, df_test]:
    # mean imputation for continuous variables
    for feature in ["age_of_driver", "annual_income", "claim_est_payout", "age_of_vehicle"]:
        feature_mean = df_train.loc[:, feature].mean(skipna=True)
        df[feature].fillna(int(feature_mean), inplace=True)

    # mode imputation for categorical variables
    for feature in ["marital_status", "witness_present_ind", "zip_code"]:
        feature_mode = df_train.loc[:, feature].mode(dropna=True)
        df[feature].fillna(feature_mode.values[0], inplace=True)
for df in [df_train, df_val, df_test]:
    df.drop(columns=["claim_date", "claim_day_of_week", "vehicle_color"], inplace=True)


zip_code_database = pd.read_csv("E:/S2_network/project4/pythonProject3/data/external/zip_code_database.csv")
latitude_and_longitude_lookup = {
    row.zip: (row.latitude, row.longitude) for row in zip_code_database.itertuples()
}

for df in [df_train, df_val, df_test]:
    df["latitude"] = df["zip_code"].apply(lambda x: latitude_and_longitude_lookup[x][0])
    df["longitude"] = df["zip_code"].apply(lambda x: latitude_and_longitude_lookup[x][1])

for df in [df_train, df_val, df_test]:
    df.drop(columns=["zip_code"], inplace=True)

df_train.to_csv("E:/S2_network/project4/pythonProject3/data/processed/train.csv", index=False)
df_val.to_csv("E:/S2_network/project4/pythonProject3/data/processed/val.csv", index=False)
df_test.to_csv("E:/S2_network/project4/pythonProject3/data/processed/test.csv", index=False)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from modeling import xgb_pipeline_random_feature

df_train = pd.read_csv("E:/S2_network/project4/pythonProject3/data/raw/train.csv")



def plot_categorical_feature_by_target(df, feature, target="fraud"):
    totals = df[feature].value_counts().sort_index()
    fig = sns.catplot(data=df, x=feature, hue=target, kind="count", order=totals.index).set(
        title=f"{target} by {feature}")

    # Rotate the x-axis labels if they are too long
    if any(isinstance(x, str) and len(x) >= 20 for x in totals.index):
        plt.xticks(rotation=90)

    # Add percentages
    bars = [p.get_height() for p in fig.ax.patches]
    patch = [p for p in fig.ax.patches]
    num_feature_categories = df[feature].nunique()
    num_target_categories = df[target].nunique()
    if num_feature_categories < 4:
        if num_feature_categories == 2:
            offset = 0.1
        elif num_feature_categories == 3:
            offset = 0.18
        elif num_feature_categories == 4:
            offset = 0.25

        for i in range(num_feature_categories):
            total = totals.iloc[i]
            for j in range(num_target_categories):
                bar_idx = j * num_feature_categories + i
                percentage = f"{bars[bar_idx] / total * 100:.1f}%"

                x = patch[bar_idx].get_x() + patch[bar_idx].get_width() / 2 - offset
                y = patch[bar_idx].get_y() + patch[bar_idx].get_height() + 3
                fig.ax.annotate(percentage, (x, y), size=12)


def plot_numerical_feature_by_target(df, feature, target="fraud"):
    # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    figure, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    figure.suptitle(f"{target} by {feature}")

    # Assign a graph to each ax
    sns.boxplot(data=df, x=feature, y=target, ax=ax_box)
    sns.histplot(data=df, x=feature, hue=target, ax=ax_hist)

    # Remove x axis label for the boxplot
    ax_box.set(xlabel="")


# Display summary statistics for numerical columns
print("Summary statistics for numerical columns:")
print(df_train.describe())

# Display top 10 columns with the most missing values
print("\nTop 10 columns with the most missing values:")
print(df_train.isnull().sum(axis=0).sort_values(ascending=False)[:10])


# Display the count of each unique value in the "fraud" column
print(df_train["fraud"].value_counts())

# Remove rows where the "fraud" column has a value of -1
df_train = df_train[df_train["fraud"] != -1]

# Convert the "fraud" column to string type
df_train["fraud"] = df_train["fraud"].astype(str)

for feature in ["age_of_driver", "annual_income", "safty_rating", "past_num_of_claims"]:
    plot_numerical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot

# Loop through categorical features and generate plots
for feature in ["gender", "marital_status", "high_education_ind", "address_change_ind", "living_status"]:
    plot_categorical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot

# pd.crosstab(index=df_train["zip_code"], columns=df_train["fraud"], normalize="index")["1"].describe()
# Compute the cross-tabulation of fraud cases by zip code
cross_tab = pd.crosstab(index=df_train["zip_code"], columns=df_train["fraud"], normalize="index")

# Extract the proportions of fraud cases labeled as "1" and compute descriptive statistics
fraud_proportions = cross_tab["1"]
fraud_proportions_description = fraud_proportions.describe()

# Print the descriptive statistics
print(fraud_proportions_description)

print("Mean:", fraud_proportions_description["mean"])
print("Standard Deviation:", fraud_proportions_description["std"])

zip_code_counts = df_train["zip_code"].value_counts()
print(zip_code_counts)

df_train["claim_date"] = pd.to_datetime(df_train["claim_date"])
df_train["claim_year"] = df_train["claim_date"].dt.year
df_train["claim_month"] = df_train["claim_date"].dt.month

for feature in ["claim_year", "claim_month", "claim_day_of_week", "accident_site", "witness_present_ind", "channel", "policy_report_filed_ind"]:
    plot_categorical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot

# Loop through numerical features and generate plots
for feature in ["liab_prct", "claim_est_payout"]:
    plot_numerical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot
import matplotlib.pyplot as plt

# Loop through categorical features and generate plots
categorical_features = ["vehicle_category", "vehicle_color"]
for feature in categorical_features:
    plot_categorical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot

# Loop through numerical features and generate plots
numerical_features = ["age_of_vehicle", "vehicle_price", "vehicle_weight"]
for feature in numerical_features:
    plot_numerical_feature_by_target(df_train, feature)
    # plt.show()  # Display the plot

# Your preprocessing steps
df_train["fraud"] = df_train["fraud"].astype(int)
df_train = pd.get_dummies(df_train, drop_first=True, columns=["gender", "living_status", "accident_site", "channel", "vehicle_category"])

# Exclude non-numeric columns from the correlation matrix
non_numeric_columns = ["claim_number", "claim_year", "claim_month", "claim_day_of_week"]
corr_columns = ~df_train.columns.isin(non_numeric_columns)

# Convert remaining columns to numeric format
df_train_numeric = df_train.loc[:, corr_columns].apply(pd.to_numeric, errors='coerce')

# Create correlation matrix
corr_matrix = df_train_numeric.corr().round(3)

# Create and display heatmap
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Close the figure to release memory
plt.close()

import pickle

with open("best_model.pickle", "wb") as f:
    pickle.dump(xgb_pipeline_random_feature, f)

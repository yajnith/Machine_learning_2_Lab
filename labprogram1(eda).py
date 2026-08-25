import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("DATASET.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.dtypes)

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

print("\nMissing values:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(exclude=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

for column in numeric_columns:
    values = df[column].to_numpy()
    print(f"\n{column}")
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("Standard Deviation:", np.std(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))

df[numeric_columns].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distribution of Numerical Variables")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
df[numeric_columns].boxplot()
plt.title("Box Plot of Numerical Variables")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

if len(numeric_columns) >= 2:
    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df[x_column],
        df[y_column],
        alpha=0.7,
        edgecolors="black"
    )

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{x_column} vs {y_column}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts())

print("\nFinal dataset shape:", df.shape)
print("\nEDA completed successfully!")

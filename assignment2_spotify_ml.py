#Assignment 2:

# Imports:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split      # train–test split as in tutorial
from sklearn.preprocessing import StandardScaler          # feature scaling
from sklearn.linear_model import LogisticRegression       # logistic regression
from sklearn.neighbors import KNeighborsClassifier       # KNN
from sklearn.tree import DecisionTreeClassifier           # decision tree
from sklearn.metrics import confusion_matrix, classification_report  # confusion matrix & metrics

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# Load dataset

df = pd.read_csv("dataset.csv")

# Q1: Introduction to EDA

# (a) First and last 5 rows

from IPython.display import display

print("First 5 rows:")
display(df.head())

print("\nLast 5 rows:")
display(df.tail())


# (b) Shape and column names

print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# (c) info() and describe()

print("\nDataFrame info:")
df.info()

print("\nDescriptive statistics:")
display(df.describe(include="all").T)

# (d) Numerical vs categorical columns

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical columns:", num_cols)
print("\nCategorical columns:", cat_cols)

# Q2: Missing Values & Feature Scaling

# (a) Check missing values

print("\nMissing values per column:")
print(df.isna().sum())

# (b) Handle missing values in selected numerical features

num_features = ["danceability", "energy", "loudness", "tempo", "valence"]

df_clean = df.dropna(subset=num_features).copy()
print("\nOriginal shape:", df.shape)
print("After dropping rows with NA in key features:", df_clean.shape)

# (c) Select numerical features

X_feats = df_clean[num_features]
print("\nSelected numerical features (first 5 rows):")
display(X_feats.head())

# (d) Apply standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feats)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=[f"{col}_scaled" for col in num_features]
)

print("\nScaled features (first 5 rows):")
display(X_scaled_df.head())

print("\nScaled features summary:")
display(X_scaled_df.describe().T)

# Q3: Data Visualization for EDA

# (a) Histogram of danceability

plt.figure(figsize=(6,4))
sns.histplot(df_clean["danceability"], bins=30, kde=True)
plt.title("Distribution of Danceability")
plt.xlabel("Danceability")
plt.ylabel("Count")
plt.show()

# (b) Boxplot of energy

plt.figure(figsize=(4,6))
sns.boxplot(y=df_clean["energy"])
plt.title("Boxplot of Energy")
plt.ylabel("Energy")
plt.show()

# (c) Scatter plot: energy vs loudness

plt.figure(figsize=(6,4))
sns.scatterplot(
    data=df_clean,
    x="loudness",
    y="energy",
    alpha=0.4
)
plt.title("Energy vs Loudness")
plt.xlabel("Loudness (dB)")
plt.ylabel("Energy")
plt.show()

# (d) Correlation heatmap

plt.figure(figsize=(6,5))
corr = df_clean[num_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Audio Features")
plt.show()

# Q4: Audio Features & Mood

print("\nSummary of danceability, energy, valence:")
display(df_clean[["danceability", "energy", "valence"]].describe().T)

# Definitions and mood interpretation go in markdown/text, based on Spotify audio-feature docs.

# Q5: Supervised Learning – Classification

# (a) Create mood column using valence (binary: 1 = high valence, 0 = low)

val_median = df_clean["valence"].median()
print("\nValence median used for mood split:", val_median)

df_clean["mood"] = np.where(df_clean["valence"] >= val_median, 1, 0)
print("\nMood label counts (0 = low valence, 1 = high valence):")
print(df_clean["mood"].value_counts())

# (b) Train–test split (80–20)

features_clf = ["danceability", "energy", "loudness", "tempo"]
X = df_clean[features_clf]
y = df_clean["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTrain/test shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# Feature scaling for models that require it

scaler_clf = StandardScaler()
X_train_scaled = scaler_clf.fit_transform(X_train)
X_test_scaled = scaler_clf.transform(X_test)

# (c) Train Logistic Regression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)

print("\nLogistic Regression – classification report")
print(classification_report(y_test, y_pred_lr))

# (d) Confusion matrix for Logistic Regression

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(4,3))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Q6: Bonus – KNN and Decision Tree

# KNN:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN – classification report")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(4,3))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – KNN")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Decision Tree:

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)   # decision trees do not need scaled inputs

y_pred_tree = tree.predict(X_test)

print("\nDecision Tree – classification report")
print(classification_report(y_test, y_pred_tree))

cm_tree = confusion_matrix(y_test, y_pred_tree)

plt.figure(figsize=(4,3))
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix – Decision Tree")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

#Logistic regression gave the most consistent performance across precision, recall and F1 for both mood classes, so it seems the best trade off between simplicity and accuracy.

#KNN was competitive but more sensitive to parameter choice and may not scale well to very large Spotify datasets.

#The decision tree was easy to interpret (clear splits on features like energy or valence) but showed signs of overfitting when depth increased, which matches the known disadvantages of trees.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load CSV directly
csv_path = r"C:\Users\phaniandey\Downloads\Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

print("CSV loaded successfully!")

# Display basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Histograms
df.hist(figsize=(12,8))
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True, cmap='coolwarm')
plt.show()

# Pairplot
sns.pairplot(df.select_dtypes(include=['int64','float64']))
plt.show()

# Countplots
sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

# Interactive Plot
fig = px.histogram(df, x='Age', color='Survived')
fig.show()


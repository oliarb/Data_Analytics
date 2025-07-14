import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set visualization styles
sns.set(style='whitegrid', palette='muted', color_codes=True)

# Data Loading
file_path = r"C:\Users\oliar\OneDrive\Documents\Data_Analysis\Databases\vgsales.txt"
df = pd.read_csv(file_path, encoding='utf-8')

# Display first few rows as a preview (output will be generated when run)
df.head()

# Data Cleaning and Preprocessing

# Let's take a look at the data info to understand missing values and data types
df.info()

# Convert Year to a numeric type if not already; if there are parsing errors, coerce them into NaN
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop rows with missing Year values for simplicity in this analysis
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)  # Convert back to int after removing NaNs

# Strip whitespace from string columns to avoid inconsistency
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    df[col] = df[col].str.strip()

print('Data cleaning and preprocessing complete.')

# Show summary statistics for numeric columns
df.describe()

# Exploratory Data Analysis

# 1. Distribution of Global Sales using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Global_Sales'], bins=30, kde=True)
plt.title('Distribution of Global Sales')
plt.xlabel('Global Sales (millions)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Count plot for Platform (Pie charts are less effective for complex category distributions, so using countplot instead)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Platform', order=df['Platform'].value_counts().index)
plt.title('Count of Video Game Platform')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Count plot for Publisher (Pie charts are less effective for complex category distributions, so using countplot instead)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Publisher', order=df['Publisher'].value_counts().index)
plt.title('Count of Video Game Publisher')
plt.xlabel('Publisher')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Count plot for Genres (Pie charts are less effective for complex category distributions, so using countplot instead)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Genre', order=df['Genre'].value_counts().index)
plt.title('Count of Video Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap (restricting to numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()

# 4. Pair Plot of selected numeric features
cols_for_pairplot = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sns.pairplot(df[cols_for_pairplot], diag_kind='kde')
plt.suptitle('Pair Plot of Sales Regions', y=1.02)
plt.show()

# Predictive Modeling

# In this section, we attempt to predict Global Sales using the sales from different regions.
# We use a linear regression model. Note: It is often useful to engineer features and consider categorical variables. 

# For simplicity, we will use NA_Sales, EU_Sales, JP_Sales, and Other_Sales as predictors.
features = ['Platform', 'Publisher', 'Genre']
target = 'Global_Sales'

# Prepare the data for modeling
X = df[features]
y = df[target]

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Evaluate the model using R2 score as a metric
r2 = r2_score(y_test, y_pred)
print(f'R2 Score for the Linear Regression model: {r2:.2f}')

# Optional: Visualize the comparison between actual and predicted global sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual vs Predicted Global Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()
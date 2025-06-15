
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# %%
df=pd.read_excel('marketing_campaign1.xlsx',sheet_name="marketing_campaign")
df

# %%
df.head()

# %%
df.tail()

# %%
df.isnull().sum()

# %%
# --- Step 1: Handle Missing Values ---
# Fill Income missing values with median
df['Income'].fillna(df['Income'].median(), inplace=True)

# %%
df.isnull().sum()

# %%
df.info()

# %%
# Convert Year_Birth to Age
from datetime import datetime
df['Age'] = datetime.now().year - df['Year_Birth']

# %%
columns_to_drop = [ 'Year_Birth']
df.drop(columns=columns_to_drop, inplace=True)

# %%
df.head()

# %%
# One hot encoding
categorical_cols = ['Education', 'Marital_Status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# %%
print(df.columns)

# %%
df_encoded.head(10)

# %%
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove("ID")  # ID is not a feature

# %%
# Function to remove outliers using IQR method
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

# %%
# Apply the function
df_clean = remove_outliers_iqr(df_encoded, numeric_cols)

# %%
# Check the shape before and after
print("Original shape:", df_encoded.shape)
print("After outlier removal:", df_clean.shape)

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df_encoded, y='Income')
plt.title("Before Removing Outliers")

# %%
plt.subplot(1, 2, 2)
sns.boxplot(data=df_clean, y='Income')
plt.title("After Removing Outliers")
plt.tight_layout()
plt.show()

# %%
# --- Step 3: Exploratory Data Analysis ---
# Summary statistics
summary_stats = df.describe()

# %%
summary_stats

# %%
print("Before:\n", df_encoded['Income'].describe())
print("After:\n", df_clean['Income'].describe())

# %%
# Visualizations
plt.figure(figsize=(15, 6))
sns.histplot(df['Income'], kde=True, bins=30)
plt.title('Income Distribution')
plt.show()

# %% [markdown]
# Highly Right-Skewed Distribution
# The majority of customers have incomes concentrated between ~$20,000 to $80,000.
# 
# A long right tail extends up to $600,000+, but very few individuals fall in this high-income range.

# %%
plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='Education')
plt.title('Education Level Distribution')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# This bar chart displays the number of customers across different education levels.
# 
# Majority of Customers Are Graduates
# The 'Graduation' category dominates with over 1,000 customers.
# 
# This indicates that most of the customer base is moderately educated.

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['MntWines', 'MntMeatProducts', 'MntFishProducts']])
plt.title('Spending Distribution on Key Products')
plt.show()

# %% [markdown]
# This boxplot visualizes customer spending on three product categories: Wine, Meat, and Fish.
# 
#  MntWines – Highest and Widest Spending Range
# Median spending is higher compared to meat and fish.
# 
# A wide interquartile range (IQR) shows significant variation in customer spending.
# 
# Numerous outliers above $1250 indicate a small segment of high-value wine buyers.

# %%
# Correlation heatmap
plt.figure(figsize=(18, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# The heatmap visualizes Pearson correlation coefficients between all numerical variables in your dataset. Values range from -1 (strong negative correlation) to +1 (strong positive correlation).
# 
# Top Positive Correlations
# 
# Strong positive correlations between product categories: Customers who spend more on one product category tend to spend more on others, suggesting high-value customers or lifestyle-based preferences.
# 
# Purchasing Channels: High correlation between:
# 
# NumCatalogPurchases ↔ NumWebPurchases (0.61)
# 
# NumDealsPurchases ↔ NumWebPurchases (0.5)
# 
# Income positively correlates with:
# 
# MntWines (0.45)
# 
# MntGoldProds (0.43)
# 
# NumWebPurchases (0.40)
# 
# Negative or Low Correlations
# 
# Recency (days since last purchase) has near-zero or negative correlations with spending

# %%
# 1. Numerical vs Categorical (e.g., Income vs Response)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Response', y='Income', data=df)
plt.title('Income vs Response')
plt.show()

# %% [markdown]
# Median Income is Similar
# Both groups (responders and non-responders) have comparable median incomes, slightly above $50,000.

# %%
# 2. Numerical vs Categorical: Spending on Wine vs Response
plt.figure(figsize=(8, 5))
sns.boxplot(x='Response', y='MntWines', data=df)
plt.title('Wine Spending vs Response')
plt.show()

# %% [markdown]
# compares spending on wine (MntWines) between customers who responded to a marketing campaign (Response = 1) and those who did not (Response = 0).
# 
#  Higher Wine Spending Among Responders:Customers who spend more on wine are more likely to respond to marketing campaigns.
# 

# %%
# 3. Categorical vs Categorical: Education vs Response
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: clean column names
df.columns = df.columns.str.strip()

# Optional: convert numeric response
if df['Response'].dtype != 'object':
    df['Response'] = df['Response'].map({0: 'No', 1: 'Yes'})

# Drop missing values in those columns
df = df.dropna(subset=['Education', 'Response'])

# Plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Education', hue='Response', data=df)
plt.title('Response by Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# %% [markdown]
#  compares customer responses to a marketing campaign across different levels of education.
# 
#  Most Customers Are Graduates
# The majority of customers fall under the "Graduation" category.
# 
# Basic and 2n Cycle Education Levels Have Low Response Rates

# %%
# 4. Categorical vs Categorical: Marital Status vs Response
plt.figure(figsize=(8, 5))
sns.countplot(x='Marital_Status', hue='Response', data=df)
plt.title('Response by Marital Status')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# shows how customers with different marital statuses responded to a marketing campaign (Response: 1 = Responded, 0 = Did not respond).
# 
#  Married and Together Are the Largest Segments
# Divorced and Widowed Show Lower Engagement

# %%
# 5. Numerical vs Categorical: Number of Web Purchases vs Response
plt.figure(figsize=(8, 5))
sns.boxplot(x='Response', y='NumWebPurchases', data=df)
plt.title('Web Purchases vs Response')
plt.show()

# %% [markdown]
# shows the distribution of number of web purchases (NumWebPurchases) among customers who responded (1) vs those who did not respond (0) to a marketing campaign.
# 
# Higher Median Web Purchases for Responders
# 
#  Similar Range of Purchase Behavior

# %%
# 6. Scatter Plot: Income vs Wine Spending
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Income', y='MntWines', hue='Response', data=df)
plt.title('Income vs Wine Spending Colored by Response')
plt.show()

# %%
# Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# %%
# Total no of childrens
df['Children'] = df['Kidhome'] + df['Teenhome']

# %%
df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)

# %%
df.drop(['Z_CostContact', 'Z_Revenue','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain'], axis=1, inplace=True)

# %%
df.drop(['AcceptedCmp3'], axis=1, inplace=True)

# %%
df['Total Purchases'] = df['NumDealsPurchases'] + df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']

# %%
df.drop(['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases','NumStorePurchases'], axis=1, inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder if inverse_transform is needed later

# Optional: print encoded columns
print("Encoded columns:", list(categorical_cols))

# %%
# Convert 'Dt_Customer' to number of days since enrollment
df['Days_Since_Customer'] = (pd.to_datetime('today') - pd.to_datetime(df['Dt_Customer'])).dt.days

# drop original datetime column
df = df.drop(columns=['Dt_Customer'])

# %%
df.head()

# %% [markdown]
# **K-means Ckustering**

# %%
from sklearn.preprocessing import StandardScaler

# Select numerical features for clustering
features_for_clustering = df[['Age', 'Income', 'Total Purchases', 'Days_Since_Customer']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_for_clustering)


# %%
# K-means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# %%
# Use Elbow Method to determine optimal number of clusters
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# %%
# Plot the elbow curve
plt.figure(figsize=(8, 4))
sns.lineplot(x=list(K_range), y=inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

# %%
# Fit final KMeans with optimal k (e.g., from elbow plot, assume k=4)
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(scaled_data)

# %%
# View cluster summary
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# %%
# Visualize clusters 2D using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
plt.title('Customer Segmentation by Clusters')
plt.grid(True)
plt.show()

# %%
pca = PCA(n_components=3)
df_pca = pca.fit_transform(scaled_data)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=df["Cluster"], cmap='viridis')
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.title("3D PCA Visualization of Clusters")
plt.show()

# %%
# Evaluate clustering performance
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_data, df["Cluster"])
print("Silhouette Score:", silhouette_avg)

# %%




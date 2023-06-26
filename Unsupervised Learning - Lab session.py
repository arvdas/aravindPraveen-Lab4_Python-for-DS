#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Lab Session

# ## Learning outcomes:
# - Exploratory data analysis and data preparation for model building.
# - PCA for dimensionality reduction.
# - K-means and Agglomerative Clustering

# ## Problem Statement
# Based on the given marketing campigan dataset, segment the similar customers into suitable clusters. Analyze the clusters and provide your insights to help the organization promote their business.

# ## Context:
# - Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# - Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.

# ## About dataset
# - Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?datasetId=1546318&sortBy=voteCount
# 
# ### Attribute Information:
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month

# ### 1. Import required libraries

# In[27]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,accuracy_score,auc,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder


# ### 2. Load the CSV file (i.e marketing.csv) and display the first 5 rows of the dataframe. Check the shape and info of the dataset.

# In[3]:


df = pd.read_csv('marketing.csv')
print(df.head(5))
print("Shape of the dataset:", df.shape)
print(df.info())


# ### 3. Check the percentage of missing values? If there is presence of missing values, treat them accordingly.

# In[8]:


df = pd.read_csv('marketing.csv')
missing_percentage = (df.isna().mean() * 100).round(2)
print("Percentage of missing values:")
print(missing_percentage)


# In[9]:


df = pd.read_csv('marketing.csv')
df_filled = df.fillna(df.mean())
print(df_filled.head())


# ### 4. Check if there are any duplicate records in the dataset? If any drop them.

# In[10]:


duplicate_records = df.duplicated()
num_duplicates = duplicate_records.sum()
if num_duplicates > 0:
   
    df = df.drop_duplicates()

    print(f"{num_duplicates} duplicate record(s) found and dropped.")
else:
    print("No duplicate records found.")


print(df.head())


# ### 5. Drop the columns which you think redundant for the analysis 

# In[11]:


redundant_columns = ['ID', 'Dt_Customer']
df_dropped = df.drop(redundant_columns, axis=1)
print(df_dropped.head())


# ### 6. Check the unique categories in the column 'Marital_Status'
# - i) Group categories 'Married', 'Together' as 'relationship'
# - ii) Group categories 'Divorced', 'Widow', 'Alone', 'YOLO', and 'Absurd' as 'Single'.

# In[12]:


unique_categories = df['Marital_Status'].unique()
print("Unique Categories in 'Marital_Status':")
print(unique_categories)
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')
unique_categories_updated = df['Marital_Status'].unique()
print("\nUnique Categories in 'Marital_Status' after Grouping:")
print(unique_categories_updated)


# ### 7. Group the columns 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', and 'MntGoldProds' as 'Total_Expenses'

# In[13]:


df['Total_Expenses'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                           'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
print(df.head())


# ### 8. Group the columns 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', and 'NumDealsPurchases' as 'Num_Total_Purchases'

# In[14]:


df['Num_Total_Purchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 
                                'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
print(df.head())


# ### 9. Group the columns 'Kidhome' and 'Teenhome' as 'Kids'

# In[15]:


df['Kids'] = df[['Kidhome', 'Teenhome']].sum(axis=1)
df = df.drop(['Kidhome', 'Teenhome'], axis=1)
print(df.head())


# ### 10. Group columns 'AcceptedCmp1 , 2 , 3 , 4, 5' and 'Response' as 'TotalAcceptedCmp'

# In[16]:


df['TotalAcceptedCmp'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                             'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
df = df.drop(['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
              'AcceptedCmp4', 'AcceptedCmp5', 'Response'], axis=1)
print(df.head())


# ### 11. Drop those columns which we have used above for obtaining new features

# In[ ]:


already dropped


# ### 12. Extract 'age' using the column 'Year_Birth' and then drop the column 'Year_birth'

# In[18]:


current_year = pd.to_datetime('today').year
df['Age'] = current_year - df['Year_Birth']
df = df.drop('Year_Birth', axis=1)
print(df.head())


# ### 13. Encode the categorical variables in the dataset

# In[32]:


label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])
print(df.head())


# ### 14. Standardize the columns, so that values are in a particular range

# In[19]:


columns_to_standardize = ['Income', 'Total_Expenses', 'Num_Total_Purchases']
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
print(df.head())


# ### 15. Apply PCA on the above dataset and determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same.

# In[39]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
columns_for_pca = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                          'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                          'NumWebVisitsMonth']
df = df.dropna(subset=columns_for_pca)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=columns_for_pca)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[columns_for_pca])
pca = PCA()
pca.fit(df_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components for 90% explained variance: {n_components_90}")
print(f"Explained variance with {n_components_90} components: {cumulative_variance[n_components_90-1]}")
print(f"Number of components for 95% explained variance: {n_components_95}")
print(f"Explained variance with {n_components_95} components: {cumulative_variance[n_components_95-1]}")


# ### 16. Apply K-means clustering and segment the data (Use PCA transformed data for clustering)

# In[24]:


columns_for_pca = ['Income', 'Total_Expenses', 'Num_Total_Purchases']
df = df.dropna(subset=columns_for_pca)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=columns_for_pca)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[columns_for_pca])
pca = PCA(n_components=2)  
pca_transformed = pca.fit_transform(df_scaled)
kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_transformed)
cluster_labels = kmeans.labels_
df['Cluster'] = cluster_labels
print(df.head(


# ### 17. Apply Agglomerative clustering and segment the data (Use Original data for clustering), and perform cluster analysis by doing bivariate analysis between the cluster label and different features and write your observations.

# In[35]:


df = pd.read_csv('marketing.csv')
columns_for_clustering = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                          'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                          'NumWebVisitsMonth']
df = df.dropna(subset=columns_for_clustering)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=columns_for_clustering)
agglomerative = AgglomerativeClustering(n_clusters=3)  
cluster_labels = agglomerative.fit_predict(df[columns_for_clustering])
df['Cluster'] = cluster_labels
cluster_features = ['Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                    'MntSweetProducts', 'MntGoldProds', 'NumWebVisitsMonth']
for feature in cluster_features:
    analysis = df.groupby(['Cluster', feature]).size().reset_index(name='Count')
    print(f"Bivariate analysis between Cluster and {feature}:")
    print(analysis)
    print()


# ### Visualization and Interpretation of results

# In[ ]:





# -----
# ## Happy Learning
# -----

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
data= pd.read_csv("disney_plus_titles.csv")


# %%
data.head(15)

# %%
data.info()

# %%
data.columns.values

# %%
data.isnull().sum()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

# %%
data['release_year']=pd.to_datetime(data['release_year'],format='%Y',errors='coerce')

# %%
data=data.dropna(subset=['release_year'])

# %%
release_per_year=data['release_year'].dt.year.value_counts().sort_index()

# %%
plt.figure(figsize=(14,6))
release_per_year.plot(kind='line')
plt.title('Number of movies released per year')
plt.xlabel('Year')
plt.ylabel('Number of movies')
plt.show()


# %%
data['description']=data['description'].astype(str)


# %%
def get_sentiment(text):
    blob=TextBlob(text)
    return blob.sentiment.polarity,blob.sentiment.subjectivity

# %%
data['sentiment']=data['description'].apply(lambda x:get_sentiment(x)[0])
data['subjectivity']=data['description'].apply(lambda x:get_sentiment(x)[1])

# %%
sns.histplot(data['sentiment'],kde=True)
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show

# %%
vectorizer=TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(data['description'])

# %%
KMeans=KMeans(n_clusters=5,random_state=42)
data['cluster']=KMeans.fit_predict(X)

# %%
pca=PCA(n_components=2,random_state=42)
x_pca=pca.fit_transform(X.toarray())

# %%
plt.scatter(x_pca[:,0],x_pca[:,1],c=data['cluster'],cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# %%
print(data.head())
print(data.columns)

# %%
data['release_year']=data['release_year'].dt.year
selected_features=['release_year','rating','cluster']
data_selected=data[selected_features].dropna()
data_selected['rating']=data_selected['rating'].astype('category').cat.codes

# %%
sns.pairplot(data_selected,hue='cluster',palette='viridis',diag_kind='kde')
plt.suptitle('Pairplot of selected features',y=1.05)
plt.show()



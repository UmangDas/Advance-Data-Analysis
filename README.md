# Advanced Data Analysis

## Overview
This project applies advanced statistical and analytical methods to solve complex problems using Python. The key techniques implemented include:

- **Time Series Analysis**: Forecasting trends and seasonality in data.
- **Sentiment Analysis**: Performing text mining on unstructured data to analyze sentiment.
- **Clustering & Classification**: Using machine learning techniques to segment and recognize patterns in data.

## Dataset
- **Data Source**: `disney_plus_titles.csv`
- The dataset contains information about Disney+ titles, including release years, descriptions, and ratings.

## Getting Started
### Prerequisites
Ensure you have the following installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob
```

### Usage Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd advanced-data-analysis
   ```
3. Run the script:
   ```bash
   python analysis.py
   ```

## Implementation
### Load Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

data = pd.read_csv("disney_plus_titles.csv")
```

### Data Exploration
```python
print(data.head(15))
print(data.info())
print(data.columns.values)
print(data.isnull().sum())
```

### Time Series Analysis
```python
data['release_year'] = pd.to_datetime(data['release_year'], format='%Y', errors='coerce')
data = data.dropna(subset=['release_year'])

release_per_year = data['release_year'].dt.year.value_counts().sort_index()
plt.figure(figsize=(14,6))
release_per_year.plot(kind='line')
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()
```

### Sentiment Analysis
```python
data['description'] = data['description'].astype(str)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

data['sentiment'] = data['description'].apply(lambda x: get_sentiment(x)[0])
data['subjectivity'] = data['description'].apply(lambda x: get_sentiment(x)[1])

sns.histplot(data['sentiment'], kde=True)
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()
```

### Clustering Analysis
```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['description'])

kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(X.toarray())

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=data['cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
```

### Feature Engineering & Visualization
```python
data['release_year'] = data['release_year'].dt.year
selected_features = ['release_year', 'rating', 'cluster']
data_selected = data[selected_features].dropna()
data_selected['rating'] = data_selected['rating'].astype('category').cat.codes

sns.pairplot(data_selected, hue='cluster', palette='viridis', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.05)
plt.show()
```

## Conclusion
This project demonstrates advanced data analysis techniques, including time series forecasting, sentiment analysis, and clustering. These techniques help extract meaningful insights and patterns from complex datasets.


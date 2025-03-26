# Advanced Data Analysis

## Overview
This project applies advanced statistical and analytical methods to solve complex problems using Python. The key techniques implemented include:

- **Time Series Analysis**: Forecasting trends and seasonality in data.
- **Sentiment Analysis**: Performing text mining on unstructured data to analyze sentiment.
- **Clustering & Classification**: Using machine learning techniques to segment and recognize patterns in data.

## Dataset
- **Data Source**: `disney_plus_titles.csv`
- The dataset contains information about Disney+ titles, including release years, descriptions, and ratings.

## Key Steps
### 1. Data Exploration
- Load and inspect the dataset.
- Identify and handle missing values.
- Convert data types for better analysis.
- Perform basic statistical analysis.

### 2. Time Series Analysis
- Convert release year to a proper datetime format.
- Analyze trends in movie releases over time.
- Generate a line plot to visualize the number of movies released per year.

### 3. Sentiment Analysis
- Process text data from movie descriptions.
- Apply sentiment analysis to determine polarity (positive or negative sentiment).
- Visualize sentiment distribution using histograms.

### 4. Clustering & Classification
- Extract text features using TF-IDF vectorization.
- Apply K-Means clustering to group similar content.
- Reduce dimensionality using PCA for visualization.
- Create scatter plots to display clustering results.
- Encode categorical variables and analyze feature relationships.

## Tools & Libraries
Ensure you have the following installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob
```

## Implementation Highlights
### Time Series Analysis
- Converted release years to datetime format.
- Dropped missing values.
- Created a line chart showing trends in movie releases over time.

### Sentiment Analysis
- Converted descriptions to text format.
- Used TextBlob to extract sentiment polarity and subjectivity.
- Displayed sentiment distribution using a histogram.

### Clustering Analysis
- Used TF-IDF vectorization to process text data.
- Applied K-Means clustering with 5 clusters.
- Reduced dimensions with PCA and visualized the clusters.

### Feature Engineering & Visualization
- Encoded categorical variables.
- Created a pairplot to visualize feature relationships.

## Conclusion
This project demonstrates advanced data analysis techniques, including time series forecasting, sentiment analysis, and clustering. These techniques help extract meaningful insights and patterns from complex datasets.


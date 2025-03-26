# Advanced Data Analysis

## Getting Started
This project explores advanced data analysis techniques using Python. It includes time series analysis, sentiment analysis, and clustering techniques to extract meaningful insights from structured and unstructured data.

## Prerequisites
Before starting, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn textblob
  ```

## Overview
In this analysis, we apply various statistical and machine learning techniques to solve complex data problems:

- **Time Series Analysis**: Identifying trends and seasonality in data.
- **Sentiment Analysis**: Using text mining to analyze movie descriptions.
- **Clustering & Classification**: Implementing machine learning techniques to segment and recognize patterns in data.

## Dataset
- **Data Source**: `disney_plus_titles.csv`
- The dataset contains information on Disney+ titles, including release years, descriptions, and ratings.

## Key Steps
### 1. Data Exploration
- Load and inspect the dataset.
- Handle missing values and ensure proper data types.
- Perform basic descriptive statistics.

### 2. Time Series Analysis
- Convert the release year to a datetime format.
- Analyze trends in movie releases over time.
- Visualize the number of movies released per year using a line plot.

### 3. Sentiment Analysis
- Process text data from movie descriptions.
- Apply sentiment analysis to determine polarity (positive/negative sentiment) and subjectivity.
- Use histograms to visualize sentiment distribution.

### 4. Clustering & Classification
- Extract text features using TF-IDF vectorization.
- Implement K-Means clustering to group similar content.
- Reduce dimensions using PCA for visualization.
- Create scatter plots to display clustering results.
- Encode categorical variables and analyze feature relationships.

## Implementation Highlights
### Time Series Analysis
- Converted release years to datetime format.
- Removed missing values.
- Created a line chart to show trends in movie releases over time.

### Sentiment Analysis
- Converted descriptions to a text format.
- Used TextBlob to extract sentiment polarity and subjectivity.
- Displayed sentiment distribution with a histogram.

### Clustering Analysis
- Applied TF-IDF vectorization for text processing.
- Used K-Means clustering with 5 clusters.
- Reduced dimensions with PCA and visualized clusters using scatter plots.

### Feature Engineering & Visualization
- Encoded categorical variables for analysis.
- Created a pairplot to visualize feature relationships.

## Conclusion
This project demonstrates advanced data analysis techniques, including time series forecasting, sentiment analysis, and clustering. These methods help extract valuable insights and patterns from large datasets.

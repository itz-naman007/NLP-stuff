# Stock News Sentiment Analysis (SNM Project)

This project focuses on analyzing the sentiment of stock-related news headlines using a zero-shot learning approach. The goal is to categorize financial news from Twitter into different sentiment classes such as "positive", "neutral", or "negative".

## Project Objective

The main objective is to build a basic sentiment analysis pipeline for financial texts, particularly tweets related to the stock market. This is useful for understanding market trends, investor emotions, and possible stock movements.

## Dataset

- **Source**: Hugging Face `zeroshot/twitter-financial-news-sentiment`
- **Description**: A dataset containing tweets related to financial news along with sentiment labels.

## Libraries and Tools Used

- Python
- Pandas & NumPy (data handling)
- Seaborn & Matplotlib (data visualization)
- Scikit-learn (model building)
- Hugging Face Datasets Library (dataset loading)

## Steps Covered

1. **Environment Setup**  
   Installation of required libraries using pip.

2. **Data Loading**  
   The dataset is loaded directly from Hugging Face using the `load_dataset()` function.

3. **Data Preprocessing**  
   Basic text cleaning, null value handling, and exploratory data analysis.

4. **Modeling Approach**  
   Various machine learning models are considered, such as Logistic Regression, Naive Bayes, or SVM to classify sentiment.

5. **Evaluation**  
   Metrics like accuracy, confusion matrix, and classification reports are used to evaluate model performance.

6. **Visualization**  
   Graphs and plots are used to show sentiment distribution and performance comparison.

## Skills Demonstrated

- Dataset handling with Hugging Face
- Preprocessing of textual financial data
- Applying machine learning models for classification
- Evaluation of models using standard metrics

## How to Run

1. Clone the repository or download the notebook.
2. Make sure you have Python installed.
3. Install required dependencies using:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn datasets

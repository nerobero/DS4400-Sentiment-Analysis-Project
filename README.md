# DS4400-Sentiment-Analysis-Project
DS4400 Final project about sentiment analysis using Rotten Tomatoes user movie reviews
# Introduction 
This repository provides a sentiment analysis of movie reviews from this kaggle dataset: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
The goal is to be able to classify user reviews as positive, somewhat positive, neutral, somewhat negative, or negative. We thought that this would be interesting to explore because in class we only ever dealt with numerical data, so this would be our first time handling non-numerical data. 

We plan to compare and contrast different multi-class classification models such as Logistic regression model, Decision Tree Classifier model, Random forest classifier model, k-Nearest Neighbors Classifier model and Multinomial Naive Bayes model. 

Why is the approach a good approach compared with other competing methods? For example, did you find any reference for solving this problem previously? If there are, how does your approach differ from theirs?

What are the key components of my approach and results? Also, include any specific limitations.

# SetUp 
The dataset is comprised with phrases from the Rotten Romatoes dataset. Each sentence is parsed into multiple phrases by the Stanford parser. Each phrase has its own PhraseId and each sentence has its own SentenceId. The dataset is divided into 2: train.tsv and test.tsv. In both datasets, there is a PhraseId, SentenceId, and a Phrase. The difference between train.tsv and test.tsv is that train.tsv contains the phrases with their sentiment labels while test.tsv only contain phrases. 


The sentiment labels are: 
0 - negative 
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

We began by reading the training and testing data from the tsv files and named them train_data and test_data, respectively. Then we split the train_data into testing and validation sets and plan to use multi-class classification models listed above. 

# Results 
Multinomial NB: 
Best min_df: 0
Best alpha: 0.4
Best train score: 0.7463956170703575
Best validation score: 0.6323529411764706

Logistic Regression:

KNN: 

Decision Tree: 
Without pruning 
train score: 0.9528867102396514
validation score: 0.5914071510957324

Best max depth: 10
Best min samples split: 6
Best min samples leaf: 3
Val MSE: 0.8467576573112905
Train score: 0.5359316929386133
Val score: 0.5296360374215046




# Discussion 

# DS4400-Sentiment-Analysis-Project
DS4400 Final project about sentiment analysis using Rotten Tomatoes user movie reviews
# Introduction 
This repository provides a sentiment analysis of movie reviews from this Kaggle dataset: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
The goal is to be able to classify user reviews as positive, somewhat positive, neutral, somewhat negative, or negative. We thought that this would be interesting to explore because in class we only ever dealt with numerical data, so this would be our first time handling non-numerical data. Also, given how increasingly relevant natural language processing techniques are becoming prominent in user data analysis, we thought this project would serve as a nice introduction to it for the group. 

We plan to compare and contrast different multi-class classification models such as Decision Tree Classifier model, Random Forest Classifier model, k-Nearest Neighbors Classifier model, and Multinomial Naive Bayes model. We also plan to compare the Logistic Regression model, which is mainly used for binary classification, with other multi-class classification models. Since they are two different types of classification models, we will use both One-vs-Rest and One-vs-One techniques for the Logistic regression model. 

Both One-vs-Rest (OvR) and One-vs-One (OvO) classification models are heuristic methods that “leverages a binary classification algorithm for multi-class classifications”. The two are slightly different in such a way that OvR involves splitting the dataset into one binary dataset for each class while OvO splits it into one dataset for each class versus every other class. 

# SetUp 
The dataset is composed of phrases from the Rotten Tomatoes dataset from the Kaggle webpage linked above. Each sentence is parsed into multiple phrases by the Stanford parser. Each sentence has its own SentenceId, and each phrase has its own PhraseId. The dataset is divided into two table files: train.tsv and test.tsv. In both datasets, there is a PhraseId, SentenceId, and a Phrase. The difference between train.tsv and test.tsv is that train.tsv contains the phrases with their sentiment labels while test.tsv only contains phrases. 

The sentiment labels are: 
0 - negative 
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

The bar chart below visualizes the distribution of sentiment labels in our training data. We can see that the sentiment label with the most rows in our training data is neutral, while the two extremes, negative and positive reviews, appear the least number of times.

(insert image here)

We began by reading the training and testing data from the tsv files and named them train_data and test_data, respectively. Then we split the train_data into testing and validation sets and plan to use the multi-class classification models listed above. After training each model, predictions are made using the test_data and put through different metrics with predictions made using the validation set. 

For each model, we transformed our training and validation datasets using TfidfVectorizer. This tool is widely used in text analyses (which we are conducting a version of) in Python. According to sklearn documentation, raw data, or the sample dataset of sentences, need to be turned into numerical feature vectors before being fed to algorithms. The TfidfVectorizer vectorizes the sample sequence of symbols by tokenizing the strings, counting the occurrences of tokens in each document, and normalizing + weighting the extracted numerical features with diminishing importance tokens. 

For each model, we determined the best min_df value to pass through the TfidfVectorizer. Min_df value is the cut-off frequency value which is used to ignore terms with frequency strictly lower than the given threshold. 


MultinomialNB:
We performed cross-validation to determine the best combination of min_df (TfidfVectorizer) and alpha (MultinomialNB) values. This resulted in a min_df value of 0 and an alpha of 0.4, which we proceeded to use for other MultinomialNB models.


Logistic Regression:
As mentioned earlier, this model is mainly used for binary classification. To use this model to conduct multi-class classification, we used both the OneVsRestClassifier and OneVsOneClassifier objects. 

n_jobs=-1

KNN: 
n_neighbors=5, metric='euclidean'

Decision Tree: 
random_state=0


Feature Reduction:
While training the various models above, we observed very slow runtimes, particularly with more complex models such as decision trees and kNN. We hypothesized that as a result of vectorization, we are left with a very large number of features, which could be slowing down our models. Therefore, we attempted to perform feature reduction with the following two methods:
Setting the max_features parameter in the TfidfVectorizer
Stemming our training data before vectorization

# Results 

Multinomial NB: 
Training error: 0.2536043829296425
Test error: 0.36764705882352944
F1 score: 0.4904695785992418


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

From the results of our decision tree models, we observe that the model without pruning conditions results in overfitting, as the training set has much higher accuracy than the validation set. After cross-validation and pruning, although the overall accuracy of the model did not improve, we were able to fix the issue of overfitting.

Feature Reduction:

TfidfVectorizer max_features:
Max features: 1000
Training error: 0.44934640522875813
Test error: 0.44928232731000894
F1 score: 0.24441868774297687

Max features: 5000
Training error: 0.39462386261694216
Test error: 0.4119569396386006
F1 score: 0.35250998376828335

Max features: 10000
Training error: 0.3643790849673203
Test error: 0.3979879533512751
F1 score: 0.39857440177469183

From these results, we observe that reducing the number of features after vectorizing comes with the cost of a less accurate model.

Snowball Stemming:
MultinomialNB:
Training error: 0.3755847110085865
Test error: 0.41141227732923236
F1 score: 0.3597208642444395

kNN:
Training error: 0.26080513904908365
Test error: 0.3764898116109189
F1 score: 0.5021483017695451


# Discussion 

# Conclusion 
# Reference 

https://www.kdnuggets.com/2020/08/one-vs-rest-one-multi-class-classification.html 

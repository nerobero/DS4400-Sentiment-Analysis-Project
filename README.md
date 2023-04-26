# DS4400-Sentiment-Analysis-Project

# Introduction 
Sentiment analysis is relevant to today’s ongoing advancement with natural language processing using machine learning. From a business standpoint, it is helpful to understand user sentiment toward a company’s products. Culturally, it is also vital to comprehend overall sentiment on social media platforms and identify negative users or communities in these spaces. Therefore, for this project, we want to explore multi-class sentiment analysis using three different datasets that cover these sentiment analysis motivations: 

** Rotten Tomatoes Movie Review: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
** Amazon Kindle Book Review and Rating: https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis 
** Reddit Comments: https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Reddit_Data.csv 

The goal of this project is to observe the performances of different sentiment analysis models across these different datasets. We plan to compare and contrast different multi-class classification models such as Decision Tree Classifier model, Random Forest Classifier model, k-Nearest Neighbors Classifier model, and Multinomial Naive Bayes model. We also plan to compare the Logistic Regression model with other multi-class classification models using both One-vs-Rest and One-vs-One techniques. 

Both One-vs-Rest (OvR) and One-vs-One (OvO) classification models are heuristic methods that leverage a binary classification algorithm for multi-class classification. The two are slightly different in such a way that OvR involves splitting the dataset into one binary dataset for each class while OvO splits it into one dataset for each class versus every other class. 

# SetUp 

* Rotten Tomatoes Movie Review: 
The dataset is composed of phrases from the Rotten Tomatoes dataset linked above. Each sentence is parsed into multiple phrases by the Stanford parser. Each sentence has its own SentenceId, and each phrase has its own PhraseId. The dataset is divided into two table files: train.tsv and test.tsv. In both datasets, there is a PhraseId, SentenceId, and a Phrase. The difference between train.tsv and test.tsv is that train.tsv contains the phrases with their sentiment labels while test.tsv only contains phrases. 

The sentiment labels are: 
0 - negative 
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

The bar chart below visualizes the distribution of sentiment labels in our training data. We can see that the sentiment label with the most rows in our training data is neutral, while the two extremes, negative and positive reviews, appear the least number of times.

![alt text](https://github.com/nerobero/DS4400-Sentiment-Analysis-Project/blob/main/data_distribution.png)

* Amazon Kindle Book Review: 
The dataset consists of written reviews, the reviewer ID, helpfulness of the review, overall rating score of the product, and other miscellaneous information. We have decided to use the overall rating score as a synonymous feature to sentiment indices from the aforementioned dataset. The overall rating scores range from 1 to 5, 1 meaning poor and 5 meaning excellent. Because the numerical rating scores are related to how negative or positive the reviewers feel about the product, we decided it was fair to use the scores as sentiment index. 

The following bar graph shows the distribution of the overall rating scores for the entire dataset. We can see that generally, we have equal distribution across different scores. Scores of 4 and 5 have a bit higher distribution than the rest, which indicates that we have more positive reviews than negative/neutral reviews in the entire dataset. 

![alt text](https://github.com/nerobero/DS4400-Sentiment-Analysis-Project/blob/main/amazon_distribution.png)

* Reddit Comments: 
The dataset consists of two columns, the first column has cleaned tweets and comments from Reddit and the second column indicates its sentimental label. 

The bar chart below visualizes the distribution of sentiment labels in our data. We can see that the sentiment label with the most rows in our training data is positive, while negative reviews appear the least number of times.

![alt text](https://github.com/nerobero/DS4400-Sentiment-Analysis-Project/blob/main/reddit_distribution.png)

The sentiment labels are: 
-1 for negative, 0 for neutral, 1 for positive


Utilizing Jupyter Notebook and the NEU Discovery cluster, we begin by reading the respective csv/tsv files for each dataset. We then split the loaded data into testing and validation sets to use on the multi-class classification models listed above. After training each model, predictions and scores are calculated using the validation set to gauge model accuracy.

*TfidfVectorizor
For each model, we transformed our training and validation datasets using TfidfVectorizer. This tool is widely used in text analyses (which we are conducting a version of) in Python. According to sklearn documentation, raw data, or the sample dataset of sentences, need to be turned into numerical feature vectors before being fed to algorithms. The TfidfVectorizer vectorizes the sample sequence of symbols by tokenizing the strings, counting the occurrences of tokens in each document, and normalizing and weighting the extracted numerical features with diminishing importance tokens. For some models, we determined the best min_df value to pass through the TfidfVectorizer. Min_df value is the cut-off frequency value which is used to ignore terms with frequency strictly lower than the given threshold.



### MultinomialNB:
We performed cross-validation to determine the best combination of min_df (TfidfVectorizer) and alpha (MultinomialNB) values. This resulted in a min_df value of 0 and an alpha of 0.4, which we proceeded to use for other MultinomialNB models.


### Logistic Regression:
As mentioned earlier, this model is mainly used for binary classification. To use this model to conduct multi-class classification, we used both the OneVsRestClassifier and OneVsOneClassifier objects. 

n_jobs=-1

### KNN: 
n_neighbors=5, metric='euclidean'

### Decision Tree: 
First, we trained a decision tree model without pruning conditions set. Then we performed cross-validation on max_depth, min_samples_split, and min_samples_leaf to obtain the best combination of those parameters. 

random_state=0


### Feature Reduction:
While training the various models above, we observed very slow runtimes, particularly with more complex models such as decision trees and kNN. We hypothesized that as a result of vectorization, we are left with a very large number of features, which could be slowing down our models. Therefore, we attempted to perform feature reduction with the following two methods:<br />
1. Setting the max_features parameter in the TfidfVectorizer<br />
2. Stemming our training data before vectorization

# Results 

### Multinomial NB: <br />
Training error: 0.2536043829296425<br />
Test error: 0.36764705882352944<br />
F1 score: 0.4904695785992418


### Logistic Regression:

### KNN: 

### Decision Tree: <br />
Without pruning <br />
train score: 0.9528867102396514<br />
validation score: 0.5914071510957324

After cross-validation <br />
Best max depth: 10 <br />
Best min samples split: 6 <br />
Best min samples leaf: 3 <br />
Val MSE: 0.8467576573112905 <br />
Train score: 0.5359316929386133 <br />
Val score: 0.5296360374215046

From the results of our decision tree models, we observe that the model without pruning conditions results in overfitting, as the training set has much higher accuracy than the validation set. After cross-validation and pruning, although the overall accuracy of the model did not improve, we were able to fix the issue of overfitting.

### Feature Reduction:
Before feature reduction, we have 93,700 features.

TfidfVectorizer max_features:<br />
Max features: 1000<br />
Training error: 0.44934640522875813<br />
Test error: 0.44928232731000894<br />
F1 score: 0.24441868774297687

Max features: 5000<br />
Training error: 0.39462386261694216<br />
Test error: 0.4119569396386006<br />
F1 score: 0.35250998376828335<br />

Max features: 10000<br />
Training error: 0.3643790849673203<br />
Test error: 0.3979879533512751<br />
F1 score: 0.39857440177469183<br />

From these results, we observe that reducing the number of features after vectorizing comes with the cost of a less accurate model.


Snowball Stemming:

MultinomialNB:<br />
Training error: 0.3755847110085865<br />
Test error: 0.41141227732923236<br />
F1 score: 0.3597208642444395<br />

kNN:<br />
Training error: 0.26080513904908365<br />
Test error: 0.3764898116109189<br />
F1 score: 0.5021483017695451

Using Snowball stemming, we are able to reduce the number of features from 93,700 to 10,450. Additionally, in comparison to the previous method, stemming does not have as much of a negative effect on the accuracy of the model.


# Discussion 

# Conclusion 
# Reference 

https://www.kdnuggets.com/2020/08/one-vs-rest-one-multi-class-classification.html 

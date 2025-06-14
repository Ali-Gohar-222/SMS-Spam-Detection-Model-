# SMS-Spam-Detection-Model-


SMS Spam Classifier
This project is a simple yet effective machine learning model that classifies SMS messages as Spam or Ham (Not Spam).
It uses the Naive Bayes classifier along with TF-IDF vectorization to learn from text patterns.

🚀 Features
✅ Classifies SMS messages into spam or ham

✅ Trained on the classic UCI SMS Spam Collection dataset (~5,500 messages)

✅ Uses TF-IDF vectorization to convert text to numerical features

✅ Uses Multinomial Naive Bayes classifier

✅ Saves trained model and vectorizer using joblib

✅ Interactive input mode for testing custom messages


Component        	Library
Language  	      Python 3.x
Data handling	    pandas
Text processing  	NLTK (stopwords), scikit-learn (TF-IDF, Naive Bayes)
Visualization   	seaborn, matplotlib
Model saving  	  joblib


Accuracy: ~98% on test set

Precision (Spam): High (model minimizes false positives)

Recall (Spam): High (model catches most spam messages)


Improve ------

Train on a larger, more diverse dataset (e.g. Enron, extended SMS collections)




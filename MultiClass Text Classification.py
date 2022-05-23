#################################################################################################
# 			TASK 01:		
#			Pick any text dataset for multi-class classification task and perform classification using Sklearn
#			Using news group dataset avaliable in sklearn for the learning purpose 
# 			Taking only 4 classes for simplifcation 
#			
#				BY:- Muhammad Umer MS(AI)
#						21I-2024
#
###################################################################################################


###		IMPORTING THE NECESSARY LIBRARIES
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from pylab import savefig
import matplotlib as plt
import seaborn as sns
import numpy as np

### 	EXTRACTING THE DATA FROM SKLEARN FOR 4 CLASSES
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42)
docs_test = data_test.data

###		CREATING PIPELINE FOR DARA PRE-RPOCESSING AND CLASSIFICATION
text_class = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier(loss='hinge', penalty='l2',
	alpha=1e-3, random_state=42, 
	max_iter=5, tol=None)),

 ])

###		TRAINING THE MODEL (FIT METHOD)
text_class.fit(data_train.data, data_train.target)

###		PREDICTION OF THE CLASSES
predicted_classes = text_class.predict(docs_test)
###		Confusion Matrix\
print(metrics.classification_report(data_test.target, predicted_classes,
	target_names=data_test.target_names))
cf_matrix = metrics.confusion_matrix(data_test.target, predicted_classes)
### 	Visulization
ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
fig  = ax.get_figure()
fig.savefig("confusion_matrixTask01.png")

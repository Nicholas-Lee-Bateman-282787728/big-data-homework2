# Introduction to Big Data. Assignment №2. Stream Processing with Spark

Group project by:
 Abdurasul Rahimov, Niyaz Fahretdinov, Jafar Badour, Hussain Kara Fallah

# Description of the problem
The main problem we are tackling here is to classify tweets online. Basically provided we have a host the streams the tweets in real time. We want to build up a model that we can save and use whenever needed so that it can classify the stream data online and provide classification. So that for example if the keyword “innopolis” is mentioned in a considerable amount of positive tweets it may be used as an indicator of how well Innopolis is becoming a famous city. So we will try to solve this problem using spark/scala.

# Loading Training Data 
So first thing’s first, is loading the training data before our model do the training part and it is done via tokenizer.  Taking header into account.
So given a file which has id,text,label as three columns we are loading the data so that afterwards we will get a dataframe. 


   val training = spark.read.format("csv").option("header", "true").schema(customSchema).load("dataset/train.csv")



# Preprocessing and Feature extractors

HashingTF

HashingTF is a Transformer which takes sets of terms and converts those sets into fixed-length feature vectors. In text processing, a “set of terms” might be a bag of words. HashingTF utilizes the hashing trick. A raw feature is mapped into an index (term) by applying a hash function. *
We have used it to get 1000 features from our training model per sample.

CountVectorizer
CountVectorizer and CountVectorizerModel aim to help convert a collection of text documents to vectors of token counts. When an a-priori dictionary is not available, CountVectorizer can be used as an Estimator to extract the vocabulary, and generates a CountVectorizerModel. The model produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms like LDA.*
Choosing number of features
Here we have tested the number of features versus the test error and we have chosen a local minima to consider as our best hyper parameter

Note test error: we had the test data from validation set split algorithm.

Most popular tweet
Here is a pie chart of the most popular words that were broadcast during a 7 hours Nov 12

Figure represents piechart of the most common words in 7 hour window

So we have noticed that articles have the highest frequency which makes very much sense and we accept the result of this experiment.  Afterwards we have decided to take out articles from the streaming data having another piechart. 

Figure: represents the most common tweets keywords after article deletion

# Monitor stream and check results manually
After monitoring the stream for seven hours and writing to a final result file. We have checked 420 tweets manually and provided the result of classifier vs the result of humans. Having the following values for precision and recall.
Note we do not consider multi class classification we just consider as binary class either (positive or non positive)

Model
Precision %
Recall %
Decision Tree
75 
66
Fully connected Neural net
80
60
SVM
76
61

# Contribution of each team member

Please refer to our gitHub link for other type of visual form

Team member
First sprint
Second sprint
Third sprint
Niyaz Fahretdinov
Configured spark on their own cpu and connected to twitter streamline using api
(please check references)
Worked on Domain Adaptation and deleting non needed keywords such as articles
Merged with feature extractor and helped debug and maintain the system
Jafar Badour
Configured spark on their own cpu and connected to twitter streamline using api
(please check references)
* Worked with Abdurasul on feature extractor and transformation TDF
* implemented cross validation for DT
Wrote the readfrom streamline piece of code and tested it on local host provided by TA
Implemented FFN *
Helped writing report.


Abdurasul Rahimov
Configured spark on their own cpu and connected to twitter streamline using api
(please check references)
Worked with Jafar on feature extractor and transformation CountVectorizer
Implemented CV for SVM
Wrote the Mode file imported two classifiers from spark
And trained models.
Hussain Kara Fallah
Configured spark on their own cpu and connected to twitter streamline using api
(please check references)
Wrote the loading model from FileS
And testing a test file 
(Checked output manually)
Wrote report. 
Maintenance and testing for the project

# Solution and classifiers

So after we have manipulated the data with preprocessing and feature extraction we have to say we used three models

spark.ml.DecisionTree
spark.ml.SVM
FNN (from paper down in references, we have applied our own refactor so it fits the problem we are tackling)

Decision Tree

We have used spark machine learning library own Decision Tree classifier and we have done cross validation on the max depth and multiple other parameters (Please check gitHub for reference).
We had training accuracy up to 73%

Support Vector Machine

Also have been imported from spark and we used cross validation for the number of feature here yielded a best accuracy when having 75%

Fully Connected Neural Network
We have adapted the code to our own problem space and added new layer to the network. However unfortunately It takes too much time to train with a worse accuracy that a decision tree. Accuracy was low (and after a certain amount of epochs the model started to overfit and test error became higher).


How and on what data training and testing was carried out: (as we have mentioned above using validation set algorithm we are splitting the training data into two tracks training and testing and we are performing testing on the testing track). We have trained our model on Twitter Sentiment dataset. We have decided to reshape the problem to positive and negative sentiments rather than (positive, neutral, negative). 

# How code runs and manipulate data
So the file that the output file is basically the tweet comment and appended by a label (0,1) where 1 refers for positive and 0 refers to negative sentiment.
For example let's suppose we have the tweet as follows:
“4365, Hey man wassup I am feeling trampoline today, I am great and love candy”
After running this input without pre trained model we should expect the output of
“4365, Hey man wassup I am feeling trampoline today, I am great and love candy, 1”
 
 
 
 
# Performance comparison of models


Figure: Blue SVM, Green Decision Tree, Red Neural Network

We have found out that the best model in terms of training time utilization is DT.

Classifiers F1 score comparison
 
Classifier
F1 score
Decision Tree
0.702127659574468
FNN 
0.6857142857142857
SVM
0.6767883211678831
 




 
References
https://github.com/nearbydelta/ScalaNetwork
*spark.apache.org/docs
*https://index.scala-lang.org/thoughtworksinc/deeplearning.scala/any/1.0.0-M0?target=_2.11



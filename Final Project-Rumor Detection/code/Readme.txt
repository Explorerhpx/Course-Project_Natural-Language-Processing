2015:
You should extract features by Feature_label.py first, then use Model.py to train model basing on the features you have extracted.

Post_deal.py : By calling function 'feature_extract(Post)', this file could extract features from one post & Transform a post into a vector.
Feature_label.py: Train the LDA model; Exetract features from one all events and do PCA on them.
Model.py: Partition data set into training-set, validation-set and test-set; Train classifiers;Evaluate model.
---------------------------------------------------------------------------------------------------------------------------------------------------
2016:
post_deal15.py: use this python file to extract features from a post.
preporcessData.py: use this python file to preprocess the data output the input data of RNN models, including training data and test data.
twitterdata.py: get twitter data by the api of the Twitter.
RNN.py: build the RNN model and train the model by the data formed by the preprocessData.py, finally test the data prediction accuracy.
----------------------------------------------------------------------------------------------------------------------------------------------------
2017:
parse.py: parse the propogation tree and interpret the data, both in PTK and cPTK method.
training.py: implement the KNN-like and SVM method, and the summation of the distributed data.
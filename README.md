# Python challenge:

Identify the type of news based on headlines and short descriptions
https://www.kaggle.com/rmisra/news-category-dataset/home

Problem solution theoretical, Dataset understanding and Problem solution implementation:

The dataset is in Json format. Task was to analyze the dataset and find out the type of news based on headlines and description. The dataset has 31 categories, merged into 30 (merged 2 identical categories).
Dataset has total 124068 words, those having word length less than 5 were filtered out.
Converted the words into tokens and used Glove word embedding file to reduce to low dimention while preserving contexual similarity.
Performed matrix embedding of the words resulting in a matrix of tokens.
Selected input and output variables and split to training and test data.
Trained using Keras framework with TensorFlow backend.
Accuracy achieved was 42.4%. The model can he tested on other methods like GRU and LSTM

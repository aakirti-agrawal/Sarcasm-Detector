# SarcasmDetector

Following is the problem on detecting whether a headline is sarcastic or not using machine learning techniques.

-Clone the repository using "git clone https://github.com/aakirti-agrawal/SarcasmDetector.git".
-cd into the cloned folder.
A virtual environment "octo" has been created where the code can be run.
-To activate the environment, run "source octo/bin/activate".
This was the setup part.

Initially, data has been given as a json format. We need to change it into .csv format in order to work with it
and get inferences.
-run "ipython notebook" in the terminal.
A list of notebooks appear in the browser at 'localhost:8888'.
-open 'json_to_csv.ipynb' and run it. This will make csv files for train.json and test.json.
'train.csv' and 'test.csv' will be created in the same folder and are ready to use as train set and test set respectively.

From the notebooks that appear, open "sarcasm_model.ipynb" and run it.

Data was initially combined(train and test) in order to clean it and pick inferences.
Data Cleaning has been done before applying any model on the data.
Removal of unnecessary words like '@user' strings, stop words, words with length less than 2 and other words of no importance.

Each sentence has been tokenized and porter stemmer is applied in order to get the root word.
Now we have list of lists of all words occuring in all headlines.

Now, we need to convert the independent variables from text to numbers.
We have used two approached here:
1. Bag of Words - BagofWords count the occurrence of word in each sentence. Occurrence represent the importance of word. More frequency means more importance.

Each sentence has been assigned a vector of length 1000 which signifies the feature weights and we have used this output      as an input to the logistic Regression. Applying Logistic Regression helped us achieve a f1 score of 72.18%.

2. Word2Vec - Word representation represents the word in vector space so that if the word vectors are close to one another means that those words are related to one other.

Using word to vector model for xgboost classifier. The weights for each word in each sentence was calculated from origin      as a vector and each corresponding feature value for each sentence has been added and normalized in order to get the final    values of the features. No. of features selected was 200. Now, xgboost was applied on the resultant word to vector            dataframe. XGBoost increased the accuracy to 79.19%.

Generally, xgboost with word2vec gives a good performance score.

The train data was divided into 70-30 by using train-test split in Logistic Regression.
For the xgboost, we needed the same random sample, so we chose the train and test set from the main train set by using index from ytrain.

Finally, we have that xgboost is performing better on our given chosen random sampling. So, we apply xgboost classifier in our test.csv file and save the output in a csv file as 'output.csv'.

References:
https://skymind.ai/wiki/word2vec
https://medium.freecodecamp.org/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04 (Bag of Words)
https://projector.tensorflow.org/ (Word2Vec representation)

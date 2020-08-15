Run classifier with command line arguments like: 
python classify.py stance-data.csv "abortion"
or
python classify.py stance-data.csv "gay rights"

If you just run program as "python classify.py", it will use default arguments stance-data.csv and "abortion".

For each input topic, the program runs the training and testing of 2 classification models, ngram model and multi-feature model respectively, with 5-fold cross validation.
For each model, the accuracy of each iteration in cross validation and the average accuracy will be printed. 
The model with the higher average accuracy will be selected, and the model details, the accuracy, the f1 score and the top 20 features will be printed.

The model selection portion has been commented out. Concretely, commented out are the two alternative main() functions I used to automatically try the feature/classifier combination for each of the two models.

import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, chi2


# read data of certain topic from csv and store in numpy array
def read_data(filename, topic):
    f = open(filename, encoding='utf-8')
    data = pd.read_csv(f)
    return data.loc[data['topic'] == topic]


# get vector representation of selected ngram features
def ngramfeature(vectorizer, rawdata):
    posttext = np.array(rawdata['post_text'])
    X = vectorizer.fit_transform(posttext)
    feature_names = vectorizer.get_feature_names()
    return X.toarray(), feature_names


# get vector representation of selected liwc columns
def liwcfeature(rawdata, column):
    return np.array(rawdata[column])


# shuffle data indices and return 5-fold cross validation splits
def kfold_split(data):
    splits = KFold(n_splits=5, shuffle=True, random_state=1).split(data)
    return splits


def benchmark(clf):
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, pred)
    f1 = metrics.f1_score(test_y, pred)
    score = [accuracy, f1]
    return score


# code for optimal ngram model selection and classifier tuning
'''
def main():
    global train_x, train_y, test_x, test_y, feature_names, target_names
    argv = sys.argv
    if len(argv) < 2:
        print('Using default data and topic (abortion)')
        argv.append('stance-data.csv')
        argv.append('abortion')
    data = read_data(argv[1], argv[2])
    # encode labels
    labels = np.array(data['label'])
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    target_names = le.classes_
    # suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # try different feature representations
    ngram_trials = [CountVectorizer(),
                    CountVectorizer(stop_words='english'),
                    CountVectorizer(max_df=0.7),
                    CountVectorizer(max_df=0.8),
                    CountVectorizer(max_df=0.9),
                    CountVectorizer(ngram_range=(1, 2)),
                    CountVectorizer(ngram_range=(1, 3))
                    ]
    max_accuracy = 0
    opt_feature = None
    opt_classifier = None
    classifiers = []
    #for smoothing in [0.01, 0.1, 0.5, 1, 1.3, 1.5, 2]:
    for smoothing in range(75,85):
        classifiers.append(MultinomialNB(alpha=smoothing/100))
    for ker in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifiers.append(SVC(kernel=ker))
    classifiers.append(LinearSVC())
    for ngram_trial in ngram_trials:
        xtrial = ngramfeature(ngram_trial, data)
        print()
        print(ngram_trial)
        accuracy = np.zeros((len(classifiers), 5))
        for count, classifier in enumerate(classifiers):
            print(classifier)
            for i, (train, test) in enumerate(kfold_split(xtrial)):
                train_x = xtrial[train]
                train_y = Y[train]
                test_x = xtrial[test]
                test_y = Y[test]
                accuracy[count][i] = benchmark(classifier)

            print("Accuracy: %0.3f" % np.mean(accuracy[count]))
            if np.mean(accuracy[count]) > max_accuracy:
                max_accuracy = np.mean(accuracy[count])
                opt_feature = ngram_trial
                opt_classifier = classifier
    print()
    print("Optimum accuracy: %0.3f" % max_accuracy)
    print(opt_feature)
    print(opt_classifier)
'''

# code for finding optimal multi-feature model and classifier tuning,
# tried different combinations of feature types (LIWC, POS, raw ngram and tf-idf ngram)
# and classifiers (multinomial naive bayes, gaussian naive bayes, SVC with different kernels, LinearSVC)
'''
def main():
    global train_x, train_y, test_x, test_y, feature_names, target_names
    argv = sys.argv
    if len(argv) < 2:
        print('Using default data and topic (abortion)')
        argv.append('stance-data.csv')
        argv.append('abortion')
    data = read_data(argv[1], argv[2])
    # encode labels
    labels = np.array(data['label'])
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    target_names = le.classes_
    # suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # try different feature representations
    liwc_trials = ['word_count', 'words_pronom', 'words_per_sen', 'words_over_6',
                   'pos_emo', 'neg_emo', 'count_noun', 'count_verb', 'count_adj']
    combinations = [['word_count', 'words_pronom', 'words_per_sen', 'words_over_6',
                     'pos_emo', 'neg_emo', 'count_noun', 'count_verb', 'count_adj'],

                    ['pos_emo', 'neg_emo'],

                    ]
    max_accuracy = 0
    opt_feature = None
    opt_classifier = None
    classifiers = [MultinomialNB(),GaussianNB()]
    for ker in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifiers.append(SVC(kernel=ker, gamma='scale'))
    classifiers.append(LinearSVC())
    print()
    print('Trying Features without ngram')
    for liwc_trial in combinations:
        # xtrial = ngramfeature(ngram_trial, data)
        xtrial = liwcfeature(data, liwc_trial)
        print()
        print(liwc_trial)
        accuracy = np.zeros((len(classifiers), 5))
        for count, classifier in enumerate(classifiers):
            print(classifier)
            for i, (train, test) in enumerate(kfold_split(xtrial)):
                train_x = xtrial[train]
                train_y = Y[train]
                test_x = xtrial[test]
                test_y = Y[test]
                accuracy[count][i] = benchmark(classifier)

            print("Accuracy: %0.3f" % np.mean(accuracy[count]))
            if np.mean(accuracy[count]) > max_accuracy:
                max_accuracy = np.mean(accuracy[count])
                opt_feature = liwc_trial
                opt_classifier = classifier

    print("Optimum accuracy: %0.3f" % max_accuracy)
    print(opt_feature)
    print(opt_classifier)
    print("Trying multi-feature with unigram")
    classifiers = [MultinomialNB(),GaussianNB()]
    for ker in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifiers.append(SVC(kernel=ker, gamma='scale'))
    classifiers.append(LinearSVC())
    for liwc_trial in combinations[1]:
        ngram_trial = ngramfeature(CountVectorizer(), data).toarray()
        xtrial = np.concatenate([ngram_trial, liwcfeature(data, liwc_trial)], axis=1)
        print()
        print(liwc_trial)
        accuracy = np.zeros((len(classifiers), 5))
        for count, classifier in enumerate(classifiers):
            print(classifier)
            for i, (train, test) in enumerate(kfold_split(xtrial)):
                train_x = xtrial[train]
                train_y = Y[train]
                test_x = xtrial[test]
                test_y = Y[test]
                accuracy[count][i] = benchmark(classifier)

            print("Accuracy: %0.3f" % np.mean(accuracy[count]))
            if np.mean(accuracy[count]) > max_accuracy:
                max_accuracy = np.mean(accuracy[count])
                opt_feature = liwc_trial
                opt_classifier = classifier

    print("Optimum accuracy: %0.3f" % max_accuracy)
    print(opt_feature)
    print(opt_classifier)
'''

def main():
    global train_x, train_y, test_x, test_y, ngram_feature_names, multi_feature_names, target_names
    argv = sys.argv
    if len(argv) < 2:
        print('No valid input parameters. Using default data and topic (abortion)')
        print()
        argv.append(None)
        argv.append(None)
        argv[1] = 'stance-data.csv'
        argv[2] = 'abortion'
    data = read_data(argv[1], argv[2])
    # encode labels
    labels = np.array(data['label'])
    le = LabelEncoder()
    Y = le.fit_transform(labels)
    target_names = le.classes_
    # suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # define optimal models (feature + classifier) for each topic
    if argv[2] == 'abortion':
        ngram = CountVectorizer()
        clf_ngram = MultinomialNB(alpha=0.8)
        ngram_multi = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7)
        multi_feature = ['word_count', 'words_pronom', 'words_per_sen', 'words_over_6',
                         'pos_emo', 'neg_emo', 'count_noun', 'count_verb', 'count_adj']
        clf_multi = LinearSVC(penalty='l1', dual=False, random_state=24)
    else:
        ngram = CountVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.7)
        clf_ngram = SVC(gamma='auto', max_iter=1000, C=0.5, random_state=24)
        ngram_multi = TfidfVectorizer(ngram_range=(1, 2))
        multi_feature = ['word_count', 'words_pronom', 'words_per_sen', 'words_over_6',
                         'pos_emo', 'neg_emo', 'count_noun', 'count_verb', 'count_adj']
        clf_multi = LinearSVC(C=0.82, penalty='l1', dual=False, random_state=24)

    accuracy = np.zeros((2, 5))

    # running optimal ngram model
    ngram_vec, ngram_feature_names = ngramfeature(ngram, data)
    print("Running ngram model cross validation. Training data has shape " + str(ngram_vec.shape))
    for i, (train, test) in enumerate(kfold_split(ngram_vec)):
        train_x = ngram_vec[train]
        train_y = Y[train]
        test_x = ngram_vec[test]
        test_y = Y[test]
        accuracy[0][i] = benchmark(clf_ngram)[0]
        accuracy[1][i] = benchmark(clf_ngram)[1]
        print("Iteration %d/5 Accuracy: %0.3f" % (i + 1, accuracy[0][i]))
    print("Average accuracy of ngram model: %0.3f" % np.mean(accuracy[0]))
    print()
    max_accuracy = np.mean(accuracy[0])
    max_f1 = np.mean(accuracy[1])
    opt_feature = 'Raw Unigram counts'
    opt_classifier = 'Multinomial Naive Bayes, alpha = 0.8'

    # running optimal multi-feature model
    ngram, multi_feature_names = ngramfeature(ngram_multi, data)
    add_vec = liwcfeature(data, multi_feature)
    multi_vec = np.concatenate([add_vec, ngram], axis=1)
    multi_feature_names = np.concatenate([multi_feature, multi_feature_names])
    print("Running multi-feature model cross validation. Training data has shape " + str(multi_vec.shape))
    for i, (train, test) in enumerate(kfold_split(multi_vec)):
        train_x = multi_vec[train]
        train_y = Y[train]
        test_x = multi_vec[test]
        test_y = Y[test]
        accuracy[0][i] = benchmark(clf_multi)[0]
        accuracy[1][i] = benchmark(clf_multi)[1]
        print("Iteration %d/5 Accuracy: %0.3f" % (i + 1, accuracy[0][i]))
    print("Accuracy of multi-feature model: %0.3f" % np.mean(accuracy[0]))
    print()
    if np.mean(accuracy[0]) > max_accuracy:
        max_accuracy = np.mean(accuracy[0])
        max_f1 = np.mean(accuracy[1])
        opt_feature = 'LIWC features, POS tag counts and Tfidf of unigrams and bigrams'
        opt_classifier = 'LinearSVC with l1 penalty'
        print("The multi-feature model has the maximum accuracy for this topic.")
    else:
        print("The ngram model has the maximum accuracy for this topic.")

    print("The accuracy is %0.3f" % max_accuracy)
    print("The f1 is %0.3f" % max_f1)
    print('Features selected are: ' + opt_feature)
    print("Classifier selected is: " + opt_classifier)
    ch2 = SelectKBest(chi2, k=20)
    X_train = ch2.fit_transform(multi_vec, Y)
    topfeatures = [multi_feature_names[i] for i in ch2.get_support(indices=True)]
    print("Top 20 features are:\n%s" % "    ".join(topfeatures))


if __name__ == "__main__":
    main()

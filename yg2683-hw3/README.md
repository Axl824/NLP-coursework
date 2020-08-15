# HW3: Experiment with word embeddings
word2vec.py: Reads corpus, lowercases all words as pre-processing and builds vocabulary. Trains the 27 word2vec models using gensim lib, saves KeyedVectors to binary file, and calls sub-routines in evaluate helper for evaluation.

svd.py: Reads and preprocesses the corpus as above. Sub-routine co_matrix() takes in parameter window size and uses three embedded loops to calculate co-occurence matrix. Sub-routine ppmi_matrix() builds ppmi matrix from co-occurence matrix. Calls scipy.sparse.linalg.svds to perform svd on ppmi matrix and saves results to txt files.

eval_svd.py: Works with svd.py. After svd.py finishes, loads model from path and performs evaluation using the evaluate.py helper code.

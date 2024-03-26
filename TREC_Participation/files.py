from imports import *

# Load useful dictionaries #
print("Loading files...")
reddit_queries_documents = pickle.load(open("../../Reddit-ToT/data/reddit_questions_documents.pkl", 'rb'))
reddit_queries = pickle.load(open("../../Reddit-ToT/data/queries.pkl", 'rb'))
true_documents = pickle.load(open("../data/documents.pkl", 'rb'))
true_queries = pickle.load(open("../data/train_queries.pkl", 'rb'))
true_qrels = pickle.load(open("../data/train_qrel.pkl", 'rb'))
print("Finished loading files.")
# ----------------------- #


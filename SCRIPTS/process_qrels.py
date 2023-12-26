import pickle

documents = pickle.load(open("../WIKIPEDIA/wikipedia_documents.pkl", 'rb'))
document_titles = pickle.load(open("../WIKIPEDIA/wikipedia_titles.pkl", 'rb'))

splits = ["train", "val", "test"]
domains = ["movie", "book", "game", "music"]

for split in splits:
    for domain in domains:

        queries = pickle.load(open("../DATA/{}_queries.pkl".format(domain), 'rb'))
        query_titles = pickle.load(open("../DATA/{}_titles.pkl".format(domain), 'rb'))
        qrels = pickle.load(open("../DATA/{}_{}_qrels.pkl".format(split, domain), 'rb'))
        queries_documents = []


        for qid in qrels:

            query = queries[qid]
            query_title = query_titles[qid]
            did = qrels[qid]
            doc_title = document_titles[did]
            document = documents[did]

            new_query = query_title + ' ' + query
            new_document = doc_title + '. ' + document

            queries_documents.append( (new_query, new_document) )


        pickle.dump(queries_documents, open("../DATA/{}_{}_queries_documents.pkl".format(split, domain), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)






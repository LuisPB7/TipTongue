import pickle

documents = pickle.load(open("../data/documents.pkl", 'rb'))

split_documents = {}

for pid in documents:

    document = documents[pid]

    doc_first_sentence = document.split(".")[0]
    doc_tokens = document.split()
    doc_len = len(doc_tokens)
    if doc_len <= 512:
        document = ' '.join(doc_tokens)
        split_documents[pid] = document
    else:
        n_passages = doc_len//512
        #passage_choice = random.choice(list(range(n_passages)))
        for n in list(range(n_passages)):

            passage = doc_first_sentence + ' ' + ' '.join( doc_tokens[n*512:(n+1)*512]  )

            split_documents['{}-{}'.format(pid, n)] = passage

pickle.dump(split_documents, open("split_documents.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


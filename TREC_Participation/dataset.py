from files import *
from imports import *


class TrainDataset(Dataset):

    def __init__(self):

        # From reddit #
        self.reddit_queries_documents = reddit_queries_documents
        self.reddit_qids = list(self.reddit_queries_documents)
        self.reddit_queries = reddit_queries

        # From TREC (150 queries) #
        self.true_documents = true_documents
        self.true_queries = true_queries
        self.true_qrels = true_qrels

        # Merge #
        self.true_queries_documents = {}
        for qid in self.true_qrels:
            document = self.true_documents[self.true_qrels[qid]]
            self.true_queries_documents[qid] = document

        self.queries = {**self.reddit_queries, **self.true_queries}
        self.queries_documents = {**self.reddit_queries_documents, **self.true_queries_documents}
        self.qids = list(self.queries_documents)

        self.tokenizer = AutoTokenizer.from_pretrained("OpenMatch/co-condenser-large-msmarco")

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        sample = {}

        qid = self.qids[idx]
        query = self.queries[qid]
        document = self.queries_documents[qid]

        doc_first_sentence = document.split(".")[0]
        doc_tokens = document.split()
        doc_len = len(doc_tokens)
        if doc_len <= 512:
            document = ' '.join(doc_tokens)
        else:
            n_passages = doc_len//512
            passage_choice = random.choice(list(range(n_passages)))
            document = doc_first_sentence + ' ' + ' '.join( doc_tokens[passage_choice*512:(passage_choice+1)*512]  )

        encoded_query = self.tokenizer(query, truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN,
                                       padding='max_length')
        query_token = encoded_query['input_ids']
        query_attention = encoded_query['attention_mask']
        sample['queries'] = query_token.squeeze(0)
        sample['queries_attention'] = query_attention.squeeze(0)

        encoded_pos_passage = self.tokenizer(document, truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN,
                                             padding='max_length')
        pos_passage_token = encoded_pos_passage['input_ids']
        pos_passage_attention = encoded_pos_passage['attention_mask']
        sample['pos_passages'] = pos_passage_token.squeeze(0)
        sample['pos_passages_attention'] = pos_passage_attention.squeeze(0)

        return sample


class TestDataset(Dataset):

    def __init__(self, input_dict):
        # Dense vectors from sentence-transformers, to compare with dot product
        self.dictionary = input_dict
        self.ids = list(self.dictionary)
        self.tokenizer = AutoTokenizer.from_pretrained("OpenMatch/co-condenser-large-msmarco")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        passage = self.dictionary[self.ids[idx]]
        encoded_passage = self.tokenizer(passage, truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN,
                                         padding='max_length')
        passage_tokens = encoded_passage['input_ids'].squeeze(0)
        passage_attention = encoded_passage['attention_mask'].squeeze(0)
        sample = {'passages': passage_tokens, 'passages_attention': passage_attention}
        return sample

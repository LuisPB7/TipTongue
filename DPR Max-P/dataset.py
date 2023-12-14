import pickle, torch, random
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TrainDataset(Dataset):

    def __init__(self, args):

        self.train_qrels = pickle.load(open("../data/movies_train_qrels.pkl", 'rb'))  # Should map qid -> rel. doc_id
        self.doc_titles = pickle.load(
            open("../data/movies_document_titles.pkl", 'rb'))  # Should map doc_id -> Wikipedia title
        self.documents = pickle.load(
            open("../data/movies_train_documents.pkl", 'rb'))  # Should map movie doc_id -> Wikipedia document string
        self.queries = pickle.load(open("../data/movies_queries.pkl", 'rb'))  # Should map qid -> query text

        self.qids = list(self.train_qrels)
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        sample = {}

        qid = self.qids[idx]
        query = self.queries[qid]
        document = self.documents[self.qrels[qid]]

        doc_first_sentence = document.split(".")[0]
        doc_tokens = document.split()
        doc_len = len(doc_tokens)
        if doc_len <= 512:
            document = ' '.join(doc_tokens)

        # If the document is longer than 512 words, then choose a random 512-sized passage,
        # prepend it the doc title for context, and consider it the relevant document.
        # There should be better ways of training the model...
        else:
            n_passages = doc_len // 512
            passage_choice = random.choice(list(range(n_passages)))
            document = doc_first_sentence + ' ' + ' '.join(doc_tokens[passage_choice * 512:(passage_choice + 1) * 512])

        encoded_query = self.tokenizer(query, truncation=True, return_tensors='pt',
                                       max_length=self.args.query_max_seq_len,
                                       padding='max_length')
        query_token = encoded_query['input_ids']
        query_attention = encoded_query['attention_mask']
        sample['queries'] = query_token.squeeze(0)
        sample['queries_attention'] = query_attention.squeeze(0)

        encoded_pos_passage = self.tokenizer(document, truncation=True, return_tensors='pt',
                                             max_length=self.args.doc_max_seq_len,
                                             padding='max_length')
        pos_passage_token = encoded_pos_passage['input_ids']
        pos_passage_attention = encoded_pos_passage['attention_mask']
        sample['pos_passages'] = pos_passage_token.squeeze(0)
        sample['pos_passages_attention'] = pos_passage_attention.squeeze(0)

        return sample


class TestDataset(Dataset):

    def __init__(self, input_dict, args, is_query=False):
        self.dictionary = input_dict
        self.ids = list(self.dictionary)
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.is_query = is_query
        self.args = args

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        text = self.dictionary[self.ids[idx]]
        encoded_text = self.tokenizer(text, truncation=True, return_tensors='pt', max_length=self.args.query_max_seq_len
        if self.is_query else self.args.doc_max_seq_len, padding='max_length')

        text_tokens = encoded_text['input_ids'].squeeze(0)
        text_attention = encoded_text['attention_mask'].squeeze(0)
        sample = {'text': text_tokens, 'text_attention': text_attention}
        return sample

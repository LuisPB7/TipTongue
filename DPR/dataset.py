import pickle, torch, random
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TrainDataset(Dataset):

    def __init__(self, args):

        self.domains = args.domains
        if self.domains == 'all':
            self.domains = ['movie', 'book', 'game', 'music']
        else:
            self.domains = self.domains.split(",")

        self.qrels = {}
        self.queries = {}
        self.titles = {}

        for domain in self.domains:

            aux_qrels = pickle.load(open("../DATA/train_{}_qrels.pkl".format(domain), 'rb'))
            self.qrels = {**aux_qrels, **self.qrels}

            aux_queries = pickle.load(open("../DATA/{}_queries.pkl".format(domain), 'rb'))
            self.queries = {**aux_queries, **self.queries}

            aux_titles = pickle.load(open("../DATA/{}_titles.pkl".format(domain), 'rb'))
            self.titles = {**aux_titles, **self.titles}

        self.wiki_passages = pickle.load(open("../WIKIPEDIA/wikipedia_passages.pkl", 'rb'))

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        self.train_qids = list(self.qrels)
        print("Training on {} queries".format(len(self.train_qids)), flush=True)

    def __len__(self):
        return len(self.train_qids)

    def find_passages(self, docid):
        n_passages = 0
        for k in range(1000):
            try:
                _ = self.wiki_passages["{}-{}".format(docid, k)]
                n_passages += 1
            except:
                break

        return n_passages

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        sample = {}

        qid = self.train_qids[idx]
        query = self.queries[qid]

        rel_docid = self.qrels[qid]
        n_passages = self.find_passages(rel_docid)
        rand_ix = random.choice(list(range(n_passages)))

        document = self.wiki_passages["{}-{}".format(rel_docid, rand_ix)]

        encoded_query = self.tokenizer(self.titles[qid] + ' ' + query, truncation=True, return_tensors='pt',
                                       max_length=self.args.query_max_seq_len,
                                       padding='max_length')
        query_token = encoded_query['input_ids']
        query_attention = encoded_query['attention_mask']
        sample['queries'] = query_token.squeeze(0)
        sample['queries_attention'] = query_attention.squeeze(0)

        first_sentence = self.wiki_passages["{}-0".format(rel_docid)].split('.')[0]

        encoded_pos_passage = self.tokenizer(first_sentence + ' ' + document, truncation=True, return_tensors='pt',
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
        self.titles = pickle.load(open("../DATA/{}_titles.pkl".format(args.domain), 'rb'))
        self.args = args

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        text = self.dictionary[self.ids[idx]]

        if self.is_query:
            text = self.titles[self.ids[idx]] + ' ' + text

        else:
            pid = self.ids[idx]
            did = pid.split("-")[0]
            first_id = "{}-0".format(did)
            first_sentence = self.dictionary[first_id].split(".")[0]
            text = first_sentence + ' ' + text

        encoded_text = self.tokenizer(text, truncation=True, return_tensors='pt', max_length=self.args.query_max_seq_len
        if self.is_query else self.args.doc_max_seq_len, padding='max_length')

        text_tokens = encoded_text['input_ids'].squeeze(0)
        text_attention = encoded_text['attention_mask'].squeeze(0)
        sample = {'text': text_tokens, 'text_attention': text_attention}
        return sample

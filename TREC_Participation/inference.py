from model import DPR
from imports import *

documents = pickle.load(open("split_documents.pkl", 'rb'))
dev_queries = pickle.load(open("../data/dev_queries.pkl", 'rb'))

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

        passage = self.dictionary[ self.ids[idx] ]
        encoded_passage = self.tokenizer(passage, truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN, padding='max_length')
        passage_tokens = encoded_passage['input_ids'].squeeze(0)
        passage_attention = encoded_passage['attention_mask'].squeeze(0)
        sample = {'passages': passage_tokens, 'passages_attention':passage_attention}
        return sample


batch_size = 32

# Compute passages sparse representations #
dataset = TestDataset(documents)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
representations = []
model = DPR(None).cuda().eval()
model.load_state_dict(torch.load("dpr.pt", map_location='cuda'))


with torch.no_grad():
    for data in dataloader:
        dense_inputs = data['passages'].cuda()
        atts = data['passages_attention'].cuda()
        cls = model(dense_inputs, atts)
        representations.append(cls.cpu())

representations = torch.vstack(representations)
pickle.dump(representations, open("document_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Compute queries sparse representations #
dataset = TestDataset(dev_queries)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
representations = []

with torch.no_grad():
    for data in dataloader:
        dense_inputs = data['passages'].cuda()
        atts = data['passages_attention'].cuda()
        cls = model(dense_inputs, atts)
        representations.append(cls.cpu())

representations = torch.vstack(representations)
pickle.dump(representations, open("query_embeddings.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

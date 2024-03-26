import torch

def dot_product(queries, passages):

    # Two settings: either passages is (BATCH, DIM), or (BATCH, N_NEGS, DIM). One uses simple matmul, other einsum
    n_shape = len(passages.size())
    return torch.mm(queries, passages.transpose(0,1) if n_shape==2 else torch.einsum('xy,xzy->xz', (queries,passages)))

def cos_sim(queries, passages):
    n_shape = len(passages.size())
    queries = torch.nn.functional.normalize(queries, p=2, dim=1)
    passages = torch.nn.functional.normalize(passages, p=2, dim=1 if n_shape==2 else 2)
    return dot_product(queries, passages)


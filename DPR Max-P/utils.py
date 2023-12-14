import torch

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def dot_product(queries, passages):

    # Two settings: either passages is (BATCH, DIM), or (BATCH, N_NEGS, DIM). One uses simple matmul, other einsum
    n_shape = len(passages.size())
    return torch.mm(queries, passages.transpose(0,1) if n_shape==2 else torch.einsum('xy,xzy->xz', (queries,passages)))

def cos_sim(queries, passages):
    n_shape = len(passages.size())
    queries = torch.nn.functional.normalize(queries, p=2, dim=1)
    passages = torch.nn.functional.normalize(passages, p=2, dim=1 if n_shape==2 else 2)
    return dot_product(queries, passages)


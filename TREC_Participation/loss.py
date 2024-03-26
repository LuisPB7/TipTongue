from utils import *
from dataset import *

class ContrastiveLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct = cos_sim):
        """
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(ContrastiveLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query_vectors, pos_vectors):

        b_size = query_vectors.size(0)
        scores = self.similarity_fct(query_vectors, pos_vectors) * self.scale
        labels = torch.arange(b_size, dtype=torch.long).to(scores.device)

        return self.cross_entropy_loss(scores, labels)



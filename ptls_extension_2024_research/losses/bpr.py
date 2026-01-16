import torch
import torch.nn.functional as F

from ..sampling_strategies.triplet_selector import TripletSelector

class BPRLoss(torch.nn.Module):
    """
    BPR Loss
    """
    def __init__(self, triplet_selector: TripletSelector, 
                 ) -> None:
        super().__init__()
        self.triplet_selector = triplet_selector
    
    def forward(self, embeddings, ids):
        triplets = self.triplet_selector.get_triplets(embeddings, ids)
        # print("triplets:")
        # print(triplets)


        # print(embeddings.shape)
        # print(triplets.shape)
        # print(triplets[:, 0].shape)
        # print(triplets[:, 1].shape)

        positive_score = (embeddings[triplets[:, 0]] * embeddings[triplets[:, 1]]).sum(dim=1)
        negative_score = (embeddings[triplets[:, 0]] * embeddings[triplets[:, 2]]).sum(dim=1)
        return -F.logsigmoid(positive_score - negative_score).mean()
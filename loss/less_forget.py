import torch
import torch.nn.functional as F

def EmbeddingsSimilarity(feature_a, feature_b):
    return F.cosine_embedding_loss(
        feature_a, feature_b,
        torch.ones(feature_a.shape[0]).to(feature_a.device)
    )

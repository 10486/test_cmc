import torch
from typing import List, Iterable

from catalyst.callbacks.metric import LoaderMetricCallback
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def custom_cmc_score_count(
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor) -> float:

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(gallery_embeddings, gallery_labels)
    preds = torch.Tensor(knn.predict(query_embeddings))
    positive = preds == query_labels
    return (positive.sum()/len(positive)).item()


def custom_cmc_score(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor) -> float:

    return custom_cmc_score_count(gallery_embeddings, gallery_labels, query_embeddings, query_labels)

import numpy as np
from model import ClosestClassifier
import tensorflow as tf
from dataset import LMDBValidMLDataset
from cmc_score import custom_cmc_score_count
from torch.utils.data import DataLoader


graph = tf.Graph()
with graph.as_default():
    new_model = ClosestClassifier("./arc_model.h5")
    dataset = LMDBValidMLDataset("./data/gallery/", "./data/query/", "./config/augs.yml")
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)


    logits = []
    targets = []
    is_query = []
    for batch in loader:
        image = batch[0]
        logits.append(new_model.predict(np.array(image["image"])))
        targets.append(image["targets"])
        is_query.append(image["is_query"])
    logits = torch.Tensor(logits)
    targets = torch.Tensor(targets)
    is_query = torch.Tensor(is_query)


    gallery_embeddings = logits[~is_query]
    query_embeddings = logits[is_query]
    gallery_labels = targets[~is_query]
    query_labels = targets[is_query]
    print(custom_cmc_score_count(gallery_embeddings,gallery_labels, query_embeddings, query_labels))

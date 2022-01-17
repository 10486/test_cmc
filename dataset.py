from typing import Any, Callable, Dict, List, Optional
from catalyst.data.dataset.metric_learning import MetricLearningTrainDataset, QueryGalleryDataset
from transform import CustomAugmentator
import pyxis as px
from torch.utils.data import Dataset
from functools import lru_cache
from tqdm import tqdm

class LMDBTrainMLDataset(MetricLearningTrainDataset, Dataset):

    def __init__(self, data_folder, transforms_path: str, aug_mode: str="train"):
        self.data_folder = data_folder
        self.transforms_path = transforms_path
        self.transforms = CustomAugmentator().transforms(self.transforms_path, aug_mode=aug_mode)
        db = px.Reader(dirpath=self.data_folder)
        self.size = len(db)
        db.close()

    def __getitem__(self, idx):
        if not hasattr(self, "db"):
            self.db = px.Reader(dirpath=self.data_folder)

        item = {}
        tmp = self.db[idx]
        image = tmp["image"]
        label = tmp["target"]
        image_name = str(tmp["image_name"])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        item["features"] = image
        item["targets"] = label[0]
        item["image_name"] = image_name
        return item


    def __len__(self):
        return self.size

    @lru_cache(maxsize=30)
    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        full_targets = []
        for i in tqdm(self.db):
            full_targets.append(i["target"][0])

        return full_targets


class LMDBValidMLDataset(QueryGalleryDataset):
    # Возможно надо разделить аугментации галлереи и запроса
    def __init__(
        self, root_gallery: str, root_query: str,
        transforms_path: str, is_check: bool = False
    ) -> None:
        self.is_check = is_check

        self._gallery = LMDBTrainMLDataset(root_gallery, transforms_path=transforms_path, aug_mode='valid')
        self._query = LMDBTrainMLDataset(root_query, transforms_path=transforms_path, aug_mode='valid')

        self._gallery_size = len(self._gallery)
        self._query_size = len(self._query)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item method for dataset
        Args:
            idx: index of the object
        Returns:
            Dict with features, targets and is_query flag
        """
        if self.is_check:
            if idx % 2 == 0:
                tmp = self._gallery[idx//2]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
            else:
                tmp = self._query[idx//2]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
        else:
            if idx < self._gallery_size:
                tmp = self._gallery[idx]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
            else:
                tmp = self._query[idx - self._gallery_size]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
        return {
            "features": image,
            "targets": label,
            "image_name": image_name,
            "is_query": idx % 2 == 0 if self.is_check else idx >= self._gallery_size,
        }

    def __len__(self) -> int:
        """Length"""
        return self._gallery_size + self._query_size

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size

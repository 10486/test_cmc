from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
import numpy as np
import cv2
import tensorflow as tf
from keras.applications.mobilenet import \
    preprocess_input as mobilenet_preprocess_input
from keras.models import load_model
from keras import backend

def relu6(x):
    return backend.relu(x, max_value=6)

class Classifier(ABC):

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Tuple]:
        pass

class ClosestClassifier(Classifier):

    def __init__(self,
                 encoder_path: str,
                 input_shape: tuple = (224, 224),
                 custom_objects: dict = {'relu6': relu6}):
        self.encoder = self._load_metric_model(encoder_path, custom_objects=custom_objects)
        self._input_shape = input_shape
        self.graph = tf.Graph()

    def _load_metric_model(self, metric_model_path, *args, **kwargs):
        return load_model(metric_model_path, *args, **kwargs)

    def _preprocess_image(self, src: np.ndarray):
        src = cv2.resize(src, self._input_shape)
        src = mobilenet_preprocess_input(src)
        return np.array([src])

    def get_embedding(self, src: np.ndarray):
        src = self._preprocess_image(src)
        emb = self.encoder.predict(src)
        emb = emb[0]
        emb = emb / np.linalg.norm(emb)
        return emb

    def predict(self, image: np.ndarray) -> List[Tuple]:
        return self.get_embedding(image)

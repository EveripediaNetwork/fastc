#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Generator, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .interface import SentenceClassifierInterface


class CentroidSentenceClassifier(SentenceClassifierInterface):
    def __init__(
        self,
        embeddings_model: str,
        model: Dict[int, List[float]] = None,
    ):
        super().__init__(embeddings_model)
        self._centroids_by_label = None
        if model is not None:
            self._load_centroids(model)

    def train(self):
        if self._texts_by_label is None:
            raise ValueError("Dataset is not loaded.")

        centroids_by_label = {}
        for label, texts in self._texts_by_label.items():
            centroids_by_label[label] = np.mean(
                np.array(list(self.get_embeddings(
                    texts,
                    title='Generating embeddings [{}]'.format(label),
                    show_progress=True,
                ))),
                axis=0,
            )

        self._centroids_by_label = centroids_by_label

    def predict(self, texts: List[str]) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        if self._centroids_by_label is None:
            raise ValueError("Model is not trained.")

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        texts_embeddings = self.get_embeddings(texts)

        for text_embedding in texts_embeddings:
            cosine_similarities = {
                label: cosine_similarity([text_embedding], [centroid])[0][0]
                for label, centroid in self._centroids_by_label.items()
            }

            total = sum(cosine_similarities.values())
            probabilities = {
                label: value / total
                for label, value in cosine_similarities.items()
            }

            yield probabilities

    def predict_one(self, text: str) -> Dict[int, float]:
        return list(self.predict([text]))[0]

    def _load_centroids(self, centroids: Dict):
        self._centroids_by_label = {int(k): v for k, v in centroids.items()}

    def save_model(
        self,
        path: str,
        description: str = None,
    ):
        os.makedirs(path, exist_ok=True)
        model = {
            'version': 1.0,
            'model': {
                'type': 'centroids',
                'embeddings': self._embeddings_model._model.name_or_path,
                'data': {
                    key: value.tolist()
                    for key, value in self._centroids_by_label.items()
                },
            },
        }

        if description is not None:
            model['description'] = description

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(model, f, indent=4)

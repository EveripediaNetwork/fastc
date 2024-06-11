#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Generator, List

import torch
import torch.nn.functional as F

from .interface import SentenceClassifierInterface


class CentroidSentenceClassifier(SentenceClassifierInterface):
    def __init__(
        self,
        embeddings_model: str,
        model: Dict[int, List[float]] = None,
    ):
        super().__init__(embeddings_model)

        self._centroids = {}
        self._normalized_centroids = {}

        if model is not None:
            self._load_centroids(model)

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=2, dim=-1)

    def train(self):
        if self._texts_by_label is None:
            raise ValueError("Dataset is not loaded.")

        for label, texts in self._texts_by_label.items():
            embeddings = list(self.get_embeddings(
                texts,
                title='Generating embeddings [{}]'.format(label),
                show_progress=True,
            ))
            embeddings = torch.stack(embeddings)
            centroid = torch.mean(embeddings, dim=0)
            self._centroids[label] = centroid
            self._normalized_centroids[label] = self._normalize(centroid)

    def predict(
        self,
        texts: List[str],
    ) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        if self._normalized_centroids is None:
            raise ValueError("Model is not trained.")

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        texts_embeddings = self.get_embeddings(texts)
        normalized_texts_embeddings = [
            self._normalize(embedding)
            for embedding in texts_embeddings
        ]

        for text_embedding in normalized_texts_embeddings:
            dot_products = {
                label: torch.dot(text_embedding, centroid).item()
                for label, centroid in self._normalized_centroids.items()
            }

            total = sum(dot_products.values())
            scores = {
                label: value / total
                for label, value in dot_products.items()
            }

            yield scores

    def predict_one(self, text: str) -> Dict[int, float]:
        return list(self.predict([text]))[0]

    def _load_centroids(self, centroids: Dict):
        self._centroids = {
            int(label): torch.tensor(centroid)
            for label, centroid in centroids.items()
        }
        self._normalized_centroids = {
            label: self._normalize(centroid)
            for label, centroid in self._centroids.items()
        }

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
                    for key, value in self._centroids.items()
                },
            },
        }

        if description is not None:
            model['description'] = description

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(model, f, indent=4)

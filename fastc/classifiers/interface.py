#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class SentenceClassifierInterface:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._texts_by_label = None

    def load_dataset(self, dataset: List[Tuple[str, int]]):
        if not isinstance(dataset, list):
            raise TypeError("Dataset must be a list of tuples.")

        if not all(isinstance(text, str) and isinstance(label, int) for text, label in dataset):  # noqa: E501
            raise TypeError("Each tuple in the dataset must be (str, int).")

        texts_by_label = {}
        for text, label in dataset:
            if label not in texts_by_label:
                texts_by_label[label] = []
            texts_by_label[label].append(text)

        self._texts_by_label = texts_by_label

    def get_embeddings(
        self,
        texts: List[str],
        title: Optional[str] = None,
        show_progress: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        self._model.eval()
        with torch.no_grad():
            for text in tqdm(
                texts,
                desc=title,
                disable=not show_progress,
            ):
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                outputs = self._model(**inputs)

                token_embeddings = outputs.last_hidden_state[0]
                sentence_embedding = torch.mean(
                    token_embeddings,
                    dim=0,
                )
                yield sentence_embedding.numpy()

    def train(self):
        raise NotImplementedError

    def predict_one(self, text: str) -> Dict[int, float]:
        raise NotImplementedError

    def predict(self, texts: List[str]) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        raise NotImplementedError

    def save_model(self, path: str):
        raise NotImplementedError

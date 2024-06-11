#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

from .embeddings import EmbeddingsModel


class SentenceClassifierInterface:
    def __init__(self, model_name: str):
        self._embeddings_model = EmbeddingsModel(model_name)
        self._texts_by_label = None

    def load_dataset(self, dataset: List[Tuple[str, int]]):
        if not isinstance(dataset, list):
            raise TypeError('Dataset must be a list of tuples.')

        if not all(isinstance(text, str) and isinstance(label, int) for text, label in dataset):  # noqa: E501
            raise TypeError('Each tuple in the dataset must be (str, int).')

        texts_by_label = {}
        for text, label in dataset:
            if label not in texts_by_label:
                texts_by_label[label] = []
            texts_by_label[label].append(text)

        self._texts_by_label = texts_by_label

    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
        title: Optional[str] = None,
        show_progress: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        for text in tqdm(
            texts,
            desc=title,
            unit='text',
            disable=not show_progress,
        ):
            inputs = self._embeddings_model.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            outputs = self._embeddings_model.model(**inputs)

            token_embeddings = outputs.last_hidden_state[0]
            sentence_embedding = torch.mean(
                token_embeddings,
                dim=0,
            )
            yield sentence_embedding

    def train(self):
        raise NotImplementedError

    def predict_one(self, text: str) -> Dict[int, float]:
        raise NotImplementedError

    def predict(self, texts: List[str]) -> Generator[Dict[int, float], None, None]:  # noqa: E501
        raise NotImplementedError

    def save_model(self, path: str):
        raise NotImplementedError

    def push_to_hub(self, repo_id: str, **kwargs):
        kwargs['exist_ok'] = True
        kwargs['repo_type'] = 'model'
        create_repo(repo_id, **kwargs)
        config_path = '/tmp/fastc/config.json'
        self.save_model('/tmp/fastc')
        HfApi().upload_file(
            path_or_fileobj=config_path,
            path_in_repo='config.json',
            repo_id=repo_id,
            repo_type='model',
        )
        os.remove(config_path)

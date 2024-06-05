#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from huggingface_hub import hf_hub_download
from transformers import logging

from .classifiers.centroids import CentroidSentenceClassifier

logging.set_verbosity_error()


class ModelTypes:
    CENTROIDS = 'centroids'


class SentenceClassifier:
    def __new__(
            cls,
            model: str = None,
            embeddings_model: str = None,
    ):
        model_type = ModelTypes.CENTROIDS

        if embeddings_model is None:
            embeddings_model = 'deepset/tinyroberta-6l-768d'

        if model is not None:
            config = cls._get_config(model)
            if (
                'version' not in config
                or config['version'] != 1
            ):
                raise ValueError("Unsupported version.")

            model_type = config['model']['type']
            embeddings_model = config['model']['embeddings']
            model = config['model']['data']

        if model_type == ModelTypes.CENTROIDS:
            return CentroidSentenceClassifier(
                embeddings_model=embeddings_model,
                model=model,
            )
        else:
            raise ValueError("Unsupported model type.")

    @staticmethod
    def _get_config(model: str):
        if os.path.isdir(model):
            file_path = os.path.join(model, 'config.json')
        elif os.path.isfile(model):
            file_path = model
        else:
            file_path = hf_hub_download(
                repo_id=model,
                filename='config.json'
            )
        with open(file_path, 'r') as model_file:
            model = json.load(model_file)
        return model

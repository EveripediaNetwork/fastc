#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from huggingface_hub import hf_hub_download
from transformers import logging

from .classifiers.centroids import CentroidSentenceClassifier
from .template import Template

logging.set_verbosity_error()


class ModelTypes:
    CENTROIDS = 'centroids'


class SentenceClassifier:
    def __new__(
        cls,
        model: str = None,
        embeddings_model: str = None,
        model_type: str = None,
        template: str = None,
    ):
        model_data = None

        if model is not None:
            config = cls._get_config(model)
            model_config = config['model']
            model_type = model_config['type']
            model_data = model_config['data']
            embeddings_model = model_config['embeddings']

            if 'template' in model_config:
                template_text = model_config['template']['text']
                template_variables = model_config['template']['variables']
                template = Template(template_text, **template_variables)

        if embeddings_model is None:
            embeddings_model = 'deepset/tinyroberta-6l-768d'

        if model_type is None:
            model_type = ModelTypes.CENTROIDS

        if template is None:
            template = Template()

        if model_type == ModelTypes.CENTROIDS:
            return CentroidSentenceClassifier(
                embeddings_model=embeddings_model,
                model_data=model_data,
                template=template,
            )

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

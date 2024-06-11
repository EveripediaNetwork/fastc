#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoModel, AutoTokenizer


class EmbeddingsModel:
    _instances = {}

    def __new__(cls, model_name):
        if model_name not in cls._instances:
            instance = super(EmbeddingsModel, cls).__new__(cls)
            cls._instances[model_name] = instance
            instance._initialized = False
        return cls._instances[model_name]

    def __init__(self, model_name):
        if not self._initialized:
            self.model_name = model_name
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
            self._initialized = True

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

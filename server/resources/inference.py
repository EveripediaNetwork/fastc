#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import List

from fastc import SentenceClassifier


class InferenceResource:
    def __init__(
        self,
        model_names: List[str] = None,
        download_on_demand: bool = False,
    ):
        self._classifiers = {}
        if model_names:
            for model_name in model_names:
                self._classifiers[model_name] = SentenceClassifier(model_name)
        self._download_on_demand = download_on_demand

    def on_post(self, request, response):
        payload = request.media
        model_name = payload.get('model')
        text = payload.get('text')

        if model_name not in self._classifiers:
            if self._download_on_demand:
                self._classifiers[model_name] = SentenceClassifier(model_name)
            else:
                response.status = 404
                response.media = {
                    'error': f'Model {model_name} not found.',
                }
                return

        result = self._classifiers[model_name].predict_one(text)
        response.media = result

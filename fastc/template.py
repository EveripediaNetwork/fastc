#!/usr/bin/env python
# -*- coding: utf-8 -*-

class ModelTemplates:
    E5_INSTRUCT = 'Instruct: {instruction}\nQuery: {text}'
    E5_QUERY = 'query: {text}'
    E5_PASSAGE = 'passage: {text}'
    DEFAULT = '{text}'


class Template:
    def __init__(
        self,
        template: str = None,
        **kwargs,
    ):
        if template is None:
            template = ModelTemplates.DEFAULT
        self._template = template
        self._variables = kwargs

    def format(self, text: str) -> str:
        return self._template.format(text=text, **self._variables)

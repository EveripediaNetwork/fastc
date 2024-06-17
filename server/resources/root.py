#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib import metadata

import fastc


class RootResource:
    def on_get(self, _, response):
        response.media = {
            'version': metadata.version(fastc.__name__),
        }

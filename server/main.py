#!/usr/bin/env python
# -*- coding: utf-8 -*-

import falcon
import yaml
from resources.inference import InferenceResource
from resources.version import VersionResource

with open('./config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

app = falcon.App(
    cors_enable=True,
)

inference_resource = InferenceResource(
    model_names=config.get('cached_models', []),
    download_on_demand=config.get('download_on_demand', False)
)

app.add_route('/', inference_resource)
app.add_route('/version', VersionResource())

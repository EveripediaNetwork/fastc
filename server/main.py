#!/usr/bin/env python
# -*- coding: utf-8 -*-

import falcon
import yaml
from resources.inference import InferenceResource
from resources.root import RootResource

with open('./config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

app = falcon.App(
    cors_enable=True,
)

inference_resource = InferenceResource(
    model_names=config.get('cached_models', []),
    download_on_demand=config.get('download_on_demand', False)
)

app.add_route('/', RootResource())
app.add_route('/classify', inference_resource)

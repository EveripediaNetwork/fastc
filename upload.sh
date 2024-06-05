#!/bin/bash
set -e
cd $(dirname $0)
rm -rf dist/*.whl
pip install -r requirements-dev.txt
python -m build
twine upload dist/*.whl

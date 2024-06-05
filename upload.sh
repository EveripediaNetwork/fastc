#!/bin/bash
cd $(dirname $0)
rm -rf dist/*.whl
python -m build
twine upload dist/*.whl

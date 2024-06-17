#!/bin/bash
cd $(dirname $0)/..

if [ -z "$API_PROCESSES_COUNT" ]; then
    API_PROCESSES_COUNT=1
fi

python -m socketify main:app --host 0.0.0.0 --port 53256 --workers $API_PROCESSES_COUNT

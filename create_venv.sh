#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
deactivate

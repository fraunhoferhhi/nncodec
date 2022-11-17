#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements_cu11.txt
pip install -e .
deactivate
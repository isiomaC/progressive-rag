.PHONY: setup run report smoke all

PY = .venv/bin/python

setup:
	python3.12 -m venv .venv || brew install python@3.12 && python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt -r requirements-optional.txt

run:
	$(PY) scripts/run_experiments.py

report:
	$(PY) site/build_report.py

smoke:
	$(PY) scripts/smoke_test.py

all: run report

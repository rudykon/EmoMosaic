PYTHON ?= .venv/bin/python
ALIGN_PYTHON ?= .venv-p1/bin/python

.PHONY: help test download-data download-weights download-models check-weights predict evaluate explain align-smoke
help:
	@echo 'test | download-data | download-weights | download-models | check-weights | predict | evaluate | explain | align-smoke'
	@echo 'Run from the project root. Override PYTHON or ALIGN_PYTHON if needed.'
test:
	$(PYTHON) -m unittest discover -s code/src/p2p3 -p 'test_*.py'
download-data:
	$(PYTHON) scripts/download_data.py
download-weights:
	$(PYTHON) scripts/download_weights.py
download-models:
	$(PYTHON) code/src/problem1/download_models.py
check-weights:
	$(PYTHON) scripts/download_weights.py --check
predict:
	$(PYTHON) code/src/p2p3/predict.py --input-dir data/source_data/attachment3_missing_modalities/aligned --output validation/attachment3_predictions.csv --device cpu
evaluate:
	$(PYTHON) code/src/p2p3/evaluate_deployed.py --device cpu --bootstrap 5000
explain:
	$(PYTHON) code/src/p2p3/run_p3_fusion.py --stage all --device cpu --skip-figures
align-smoke:
	HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $(ALIGN_PYTHON) code/src/problem1/run.py --limit 1 --output "$(CURDIR)/validation/alignment_smoke"

.PHONY: website website-check website-serve
website:
	python3 scripts/build_pages.py
website-check: website
	python3 scripts/check_pages.py
	python3 scripts/sync_site_i18n.py --check
website-serve: website
	python3 -m http.server 8000 --directory _site --bind 127.0.0.1

.PHONY: install install-notebook check run lint notebook sensitive hooks

install:            ## app dependencies
	pip install -r requirements.txt

install-notebook:   ## app + notebook dependencies
	pip install -r requirements-notebook.txt

check:              ## verify environment, model and scaler
	python scripts/check_environment.py

run:                ## launch the Streamlit app
	bash scripts/run_app.sh

lint:               ## static checks
	pip install -q ruff && ruff check app scripts

sensitive:          ## scan for sensitive content before committing
	python scripts/check_sensitive.py

hooks:              ## block commits that contain sensitive content
	git config core.hooksPath .githooks

notebook:           ## open JupyterLab in notebooks/
	jupyter lab notebooks/

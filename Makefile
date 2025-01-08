install:
	pip install -r requirements.txt --upgrade

format:
	ruff format . 
	ruff check . --fix

run_mnist:
	python run_experiment.py mnist smi
	python run_experiment.py mnist map latest
	python run_experiment.py mnist ovi latest
	python run_experiment.py mnist svgd latest
	python run_experiment.py mnist asvgd latest

table_mnist:
	python make_results.py table mnist latest
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r environment/requirements.txt

lab:
	. .venv/bin/activate && jupyter lab

demo-cls:
	python scripts/train_sklearn.py --task classification --dataset iris --model random_forest

demo-reg:
	python scripts/train_sklearn.py --task regression --dataset diabetes --model linear

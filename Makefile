install:
	python -m pip install -r requirements.txt
run:
	uvicorn backend.app.main:app --reload
train:
	python scripts/train_model.py
test:
	python -m pytest -q
docker:
	docker compose up --build

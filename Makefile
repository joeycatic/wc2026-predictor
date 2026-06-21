.PHONY: lint test smoke simulate sync-live dashboard visualize data-status

lint:
	python -m ruff check .

test:
	python -m pytest

smoke:
	python main.py --simulations 10 --skip-player-features --seed 42

simulate:
	python main.py --simulations 10000 --skip-player-features --seed 42

sync-live:
	python main.py --sync-live-data --live-source auto

dashboard:
	streamlit run streamlit_app.py

visualize:
	python main.py --visualize-only

data-status:
	python main.py --data-status

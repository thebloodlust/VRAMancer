# Makefile pour VRAMancer

install:
	@echo "📦 Installation des dépendances..."
	pip install -r requirements.txt

dev-install:
	@echo "🧪 Installation des dépendances de dev..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	@echo "🧪 Lancement des tests..."
	pytest tests/

run:
	@echo "🚀 Lancement de VRAMancer (mode CLI)..."
	python launcher.py --config config.yaml

dashboard:
	@echo "📊 Lancement du dashboard..."
	vramancer-dashboard --port 5000

clean:
	@echo "🧹 Nettoyage..."
	find . -type d -name "__pycache__" -exec rm -r {} +

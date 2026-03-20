lite:
	@echo "📦 Construction de la version LITE (CLI only)..."
	tar -czf vramancer-lite.tar.gz \
		vramancer/ \
		vrm \
		requirements.txt \
		core/ \
		cli/ \
		utils/ \
		config.yaml \
		README.md
	@echo "✅ Archive LITE créée : vramancer-lite.tar.gz"
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
	bash scripts/vramancer-launcher.sh --mode cli

auto:
	@echo "🤖 Lancement intelligent (auto)..."
	bash scripts/vramancer-launcher.sh --mode auto


qt:
	@echo "📦 Construction du paquet .deb..."
	chmod +x Debian/postinst
	dpkg-deb --build Debian vramancer_1.0.deb
	@echo "🪟 Lancement Qt..."
	bash scripts/vramancer-launcher.sh --mode qt

tk:
	@echo "🧱 Lancement Tkinter..."
	bash scripts/vramancer-launcher.sh --mode tk

clean:
	@echo "🧹 Nettoyage..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -f vramancer_1.0.deb
	rm -f vramancer.tar.gz

deb:
	@echo "📦 Construction du paquet .deb..."
	chmod +x Debian/postinst
	dpkg-deb --build Debian vramancer_1.0.deb

verify-deb:
	@echo "🔍 Vérification du paquet .deb..."
	dpkg-deb --info vramancer_1.0.deb
	dpkg-deb --contents vramancer_1.0.deb

archive:
	@echo "📦 Création de l’archive .tar.gz..."
	tar -czf vramancer.tar.gz \
		launcher.py \
		vrm \
		requirements.txt \
		dashboard/ \
		core/ \
		utils/ \
		tests/ \
	

coverage:
	@echo "📊 Exécution de la suite de tests pour la couverture de code..."
	pytest tests/ --cov=core --cov-report=html --cov-report=term-missing -m "not (chaos or slow or integration)"

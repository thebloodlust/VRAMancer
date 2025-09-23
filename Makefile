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
	python launcher.py --mode cli

auto:
	@echo "🤖 Lancement intelligent (auto)..."
	vrm --mode auto

web:
	@echo "🌐 Lancement du dashboard web..."
	python launcher.py --mode web

qt:
	@echo "🪟 Lancement Qt..."
	python launcher.py --mode qt

tk:
	@echo "🧱 Lancement Tkinter..."
	python launcher.py --mode tk

clean:
	@echo "🧹 Nettoyage..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -f vramancer_1.0.deb
	rm -f vramancer.tar.gz

deb:
	@echo "📦 Construction du paquet .deb..."
	chmod +x debian/postinst
	dpkg-deb --build debian vramancer_1.0.deb

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
	

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

rust-build:
	@echo "🦀 Building Rust native extensions..."
	cd rust_core && cargo build --release
	@echo "🐍 Building Python wheel (maturin)..."
	cd rust_core && maturin build --release
	@echo "📦 Installing vramancer_rust into current venv..."
	pip install rust_core/target/wheels/vramancer_rust-*.whl --force-reinstall
	@echo "✅ Rust extensions built and installed"

rust-build-cuda:
	@echo "🦀 Building Rust extensions with CUDA support..."
	cd rust_core && cargo build --release --features cuda
	cd rust_core && maturin build --release --features cuda
	pip install rust_core/target/wheels/vramancer_rust-*.whl --force-reinstall
	@echo "✅ Rust + CUDA extensions built and installed"

native-dmabuf:
	@echo "🔗 Building DMA-BUF native extension..."
	gcc -shared -fPIC -O2 -Wall -o csrc/libvrm_dmabuf.so csrc/dmabuf_bridge.c
	@echo "✅ libvrm_dmabuf.so built in csrc/"

native-rebar:
	@echo "🔗 Building ReBAR mmap native extension..."
	gcc -shared -fPIC -O2 -Wall -o csrc/libvrm_rebar.so csrc/rebar_mmap.c
	@echo "✅ libvrm_rebar.so built in csrc/"

native-all: native-dmabuf native-rebar
	@echo "✅ All native C extensions built"

rust-test:
	@echo "🧪 Testing Rust extensions..."
	python scripts/test_rust_integration.py

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

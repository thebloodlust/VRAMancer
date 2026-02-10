from setuptools import setup, find_packages
from pathlib import Path
import re

README = Path(__file__).parent / "README.md"
long_desc = README.read_text(encoding="utf-8") if README.exists() else "VRAMancer"

# Single source of truth: core/__init__.py
_version_file = Path(__file__).parent / "core" / "__init__.py"
_version_match = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    _version_file.read_text(), re.M,
)
VERSION = _version_match.group(1) if _version_match else "0.2.4"

setup(
    name="vramancer",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vramancer = vramancer.main:main",
            "vramancer-api = core.production_api:main",
            "vramancer-health = core.health:main",
        ]
    },
    author="Jérémie",
    description="Run large AI models across multiple GPUs with smart memory balancing",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="MIT",
)

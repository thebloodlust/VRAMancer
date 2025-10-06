from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent / "README.md"
long_desc = README.read_text(encoding="utf-8") if README.exists() else "VRAMancer"

setup(
    name="vramancer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vramancer = vramancer.main:main",
            "vramancer-health = core.health:main"
        ]
    },
    author="Jérémie",
    description="Run large AI models across multiple GPUs with smart memory balancing",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="MIT",
)

from setuptools import setup, find_packages

setup(
    name="vramancer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "psutil",
        "pyyaml",
        "numpy",
        "torch"
    ],
    entry_points={
        "console_scripts": [
            "vramancer=vramancer:main",
            "vramancer-dashboard=dashboardcli:main"
        ]
    },
    author="Jérémie",
    description="Run large AI models across multiple GPUs with smart memory balancing",
    license="MIT",
)

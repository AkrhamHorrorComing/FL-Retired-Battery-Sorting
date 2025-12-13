from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="federated-battery-classification",
    version="1.0.0",
    author="Battery Classification Research Team",
    author_email="research@example.com",
    description="A federated learning system for battery classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/federated-battery-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="federated-learning, battery-classification, machine-learning, neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/federated-battery-classification/issues",
        "Source": "https://github.com/yourusername/federated-battery-classification",
    },
)
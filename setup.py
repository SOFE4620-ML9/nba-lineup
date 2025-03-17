from setuptools import setup, find_packages

setup(
    name="nba-lineup",
    version="0.1.0",
    description="NBA Lineup Prediction for Optimized Team Performance",
    author="",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
            "jupyter>=1.0.0",
        ],
    },
) 
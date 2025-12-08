from setuptools import setup, find_packages

setup(
    name="didynet",
    version="0.1.0",
    author="Zhe Liu",
    description="Differential Dynamic Network Inference from Longitudinal Multi-omics Data",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "dtaidistance",
        "joblib",
        "tqdm",
        "scipy",
        "statsmodels",
        "openpyxl"
    ],
    python_requires='>=3.8',
)
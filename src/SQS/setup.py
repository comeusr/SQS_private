from setuptools import setup

required = [
    "numpy",
    "tqdm",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
]

setup(
    name='SQS',
    version='1.1',
    description='SQS: Efficient Bayesian DNN Compression through Sparse Quantized Sub-distributions',
    packages=['SQS'],
    install_requires=required,
)
from setuptools import setup, find_packages

with open('VERSION') as version_file:
    version = version_file.read().strip()

extras_require_serving = [
    "lightgbm~=2.3",
    "xgboost~=0.90",
    "treelite_runtime~=2.2.2",
    "scikit-learn~=1.0.2",
]
extras_require_dev = [
    'coverage',
    'flake8',
    'flake8-mutable',
    'mypy',
    'pip-tools'
]

setup(
    name='pythie-serving',
    version=version,
    description='A GRPC server to serve model types using tensorflow-serving .proto services',
    author='wiremind data science team',
    author_email='data-science@wiremind.io',
    url="https://gitlab.cayzn.com/wiremind/data-science/pythie-neos.git",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy~=1.20.0",
        "grpcio~=1.30",
        "protobuf~=3.12",
    ],
    # pip-compile setup.py --no-emit-index-url --upgrade --rebuild
    # pip-compile setup.py --no-emit-index-url --upgrade --extra serving -o pythie-serving-requirements.txt
    extras_require={
        'serving': extras_require_serving,
        'dev': extras_require_serving,
        'all': extras_require_serving + extras_require_dev
    },
    entry_points={
        "console_scripts": [
            "pythie-serving=pythie_serving.run:run",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    ],
)

from setuptools import setup, find_packages

with open('VERSION') as version_file:
    version = version_file.read().strip()

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
    python_requires=">=3.7",
    install_requires=[
        "numpy==1.17.3",
        "grpcio==1.24.3",
        "xgboost==0.90",
        "protobuf==3.10.0"
    ],
    scripts=["bin/run.py"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    ],
)

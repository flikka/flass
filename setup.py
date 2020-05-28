from setuptools import setup

setup(
    name="flass",
    author="Kristian Flikka",
    author_email="kristian.flikka@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "click",
        "graphviz",
        "matplotlib",
        "mlflow",
        "numpy",
        "pandas",
        "pydot",
        "sklearn",
        "scikit-image",
        "tensorflow",
    ],
    setup_requires=["wheel"],
    description="Train Keras Convolutional Neural Network for image classification",
    long_description="Train Keras Convolutional Neural Network for image classification",
    entry_points={"console_scripts": ["flass=flass.cli:flass"]},
    packages=["flass"],
    license="MIT",
    url="http://flikka.net",
    version="0.1.5",
)

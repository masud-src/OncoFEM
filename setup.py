from setuptools import setup, find_packages

setup(
    name="oncofem",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scikit-image",
        "etelemetry==0.2.0",
        "nibabel",
        "meshio",
        "pybind11==2.2.4",
        "sympy",
        "scipy",
        "numpy-stl",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8"
        ]
    },
    python_requires=">=3.6",
    author="Marlon Suditsch",
    author_email="m.suditsch@outlook.com",
    description="Finite Element solver of Onco",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL 3",
        "Operating System :: OS Independent",
    ],
)

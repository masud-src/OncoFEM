from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "configparser",
        "pandas",
        "matplotlib",
        "nibabel",
        "meshio",
        "scipy",
        #"SVMTK",
        "skimage",
        "stl",
        #"dolfin",
        #"ufl"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8"
        ]
    },
    python_requires=">=3.6",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project description",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
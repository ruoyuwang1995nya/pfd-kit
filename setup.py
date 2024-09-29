import setuptools

__version__ = "1.0.2"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PFD-kit",
    version=__version__,
    author="Ruoyu Wang",
    author_email="ruoyuwang1995@gmail.com",
    description="fine-tune and distillation from pre-trained atomic models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruoyuwang1995nya/pfd-kit.git",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "dpdata",
        "pydflow>=1.6.57",
        "dargs>=0.3.1",
        "scipy",
        "lbg",
        "packaging",
        "fpop",
        "dpgui",
        "cp2kdata",
        "periodictable",
        "ase",
        "git+https://github.com/deepmodeling/dpgen2.git",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pfd = pfd.entrypoint.main:main",
        ],
    },
)


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DP-distill",
    version="0.0.1",
    author="Ruoyu Wang",
    author_email="ruoyuwang1995@gmail.com",
    description="Model fine-tune and distillation kit for DPA-2 pre-trained models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruoyuwang1995nya/dp-distill.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
	     'dpdata',
	     'pydflow>=1.6.57',
	     'dargs>=0.3.1',
	     'scipy',
	     'lbg',
	     'packaging',
	     'fpop',
	     'dpgui',
         'cp2kdata',
         'dpgen2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)

from setuptools import setup, find_packages

setup(
    name='singlecelldata',
    version='0.1.0',
    description='API for handling single-cell RNA-seq, hyperspectral, and brightfield data',
    author='Parmita Mishra',
    author_email='parm@precigenetics.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'anndata',
        'tifffile',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)

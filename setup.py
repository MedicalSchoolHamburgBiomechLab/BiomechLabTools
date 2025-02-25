from setuptools import find_packages, setup

setup(
    name='labtools',
    version='0.0.2',
    packages=find_packages(exclude=['test*']),
    install_requires=[
        'c3d >= 0.5.2',
        'h5py >= 3.8.0',
        'lxml >= 5.1.0',
        'matplotlib>=3.7.0',
        'numpy >= 1.26.4',
        'openpyxl >= 3.1.2',
        'pandas >= 2.2.1',
        'scipy >= 1.14.1',
    ],
    author='Dominik Fohrmann',
    author_email='dominik.fohrmann@gmail.com',
    description='A package for daily biomechanics lab usage and project analyses.',
    url='https://example.com',
)

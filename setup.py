from setuptools import setup, find_packages

setup(
    name='labtools',
    version='0.0.1',
    packages=find_packages(exclude=['test*']),
    install_requires=[
        'c3d >= 0.5.2',
        'lxml >= 5.1.0',
        'numpy >= 1.26.4',
        'openpyxl >= 3.1.2',
        'pandas >= 2.2.1',
        'scipy >= 1.14.1',
        'h5py >= 3.8.0'
    ],
    author='Dominik Fohrmann',
    author_email='dominik.fohrmann@gmail.com',
    description='A package for daily biomechanics lab usage and project analyses.',
    url='https://example.com',
)

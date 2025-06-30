from setuptools import setup, find_packages

def read_requirements(path):
    with open(path) as f:
        return f.read().splitlines()

setup(
    name='humancompatible.explain',
    version='0.1.0',
    description='AutoFair Explainability Toolkit',
    packages=find_packages(include=['humancompatible', 'humancompatible.*']),
    install_requires=[],  # No base install
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    zip_safe=False,
)

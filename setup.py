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
    extras_require={
        'facts': read_requirements('requirements_facts.txt'),
        'glance': read_requirements('requirements_glance.txt'),
        'all': list(set(
            read_requirements('requirements_facts.txt') +
            read_requirements('requirements_glance.txt')
        )),
    },
    entry_points={
        'console_scripts': [
            'explain-facts=humancompatible.explain.facts.main:main',
            'explain-glance=humancompatible.explain.glance.main:main'
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

from setuptools import setup, find_packages

# Read requirements.txt and store its contents in a list
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='FallingPlanet',
    version='0.2.6',
    packages=find_packages(),
    description='A PyTorch-based toolkit for machine learning and NLP tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='William Stigall',
    author_email='fallingplanetcorp@gmail.com',
    url='https://github.com/FallingPlanet/FallingPlanet',
    install_requires=required,
    python_requires='>=3.6',
)

from setuptools import setup, find_packages


# Add a link to github.
long_description = 'For more information see our '
long_description += '[github repository](https://github.com/anthliu/psgi).'

setup(
    name="psgi-acme",
    version="0.0.1",
    description='Parameterized Subtask Graph Inference.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anthony Liu',
    license='MIT',
    packages=['psgi'],
    install_requires=[
        'numpy>=1.19',   # TF 2.4 is compatible with 1.19 again
        'matplotlib',
        'torchtext',
        'sklearn',
        'gin-config',
        'dm-acme==0.2.0',
        #'dm-reverb-nightly==0.2.0.dev20201102',
        'dm-reverb',
        'tensorflow==2.4.0',
        'tensorflow_probability==0.12.2',
        'cloudpickle>=1.3',   # tfp requires 1.3+
        'jax',
        'jaxlib',  # CPU only: we won't be using jax, but ACME depends on jax
        'chex',
        'dm-sonnet>=2.0.0',
        'trfl>=1.1.0',
        'statsmodels',
        'gym==0.18.0',
        'ipdb',
        'absl-py',
        'pytest>=5.4.1',
        'pytest-pudb',
        'pytest-mockito',
        'tqdm',
        'graphviz==0.14.2',
        'pybind11==2.6.0',
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)

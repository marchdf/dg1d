try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'One-dimensional Discontinuous Galerkin code for Euler equations',
    'author': 'Marc T. Henry de Frahan',
    'license': 'GNU General Public License v3',
    'url': 'https://github.com/marchdf/dg1d',
    'download_url': 'https://github.com/marchdf/dg1d',
    'author_email': 'marchdf@umich.edu',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'sphinx', 'matplotlib'],
    'packages': ['dg1d'],
    'scripts': [],
    'name': 'dg1d'
}

setup(**config)

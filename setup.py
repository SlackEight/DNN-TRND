from setuptools import setup, find_packages

setup(
    name         = 'dnntrn',
    version      = '1.0',
    packages     = find_packages(),
    scripts      = ['utils/polar_pla.py'],
)
from setuptools import find_packages, setup

setup(
    name='appliance_energy_predictor',
    version='0.0.0',
    author='Prarthana Sewmini Ganegama',
    author_email='39-dba-0025@kdu.ac.lk',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[]
)

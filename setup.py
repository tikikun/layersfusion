from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='layersfusion',
    version='0.0.1',
    include_package_data=True,
    python_requires='>=3.10.0',
    license='MIT',
    author="Alan Dao",
    author_email='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/tikikun/layersfusion',
    keywords='',
    install_requires=requirements,
)

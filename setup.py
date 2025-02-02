from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering'
]

setup(
    name='cantileverwall',
    version='0.1.2',
    description='Library for calculating safety factors of cantilever walls',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/semihyumusak/CantileverWallSafety',
    author='Semih Yumuşak',
    author_email='semihyumusak@yahoo.com',
    license='MIT',
    classifiers=classifiers,
    keywords='cantilever wall, geotechnical engineering, safety factors, sliding, overturning, slope stability',
    packages=find_packages(),
    install_requires=['numpy', 'scipy']
)

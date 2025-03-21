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
    name='retainingwall_safety',
    version='0.1.5',
    description='Library for calculating safety factors of cantilever retaining walls',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/semihyumusak/RetainingWallSafetyEvaluator',
    author='Semih Yumuşak',
    author_email='semihyumusak@yahoo.com',
    license='MIT',
    classifiers=classifiers,
    keywords='cantilever wall, geotechnical engineering, safety factors, sliding, overturning, slope stability',
    packages=find_packages(),
    install_requires=['numpy', 'scipy']
)

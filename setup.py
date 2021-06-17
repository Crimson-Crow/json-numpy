from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='json-numpy',
    version='1.0.1a1',
    description='JSON encoding/decoding for Numpy arrays and scalars',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Crimson-Crow/json-numpy',
    author='Crimson-Crow',
    author_email='github@crimsoncrow.dev',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='json numpy serialization encoding',
    license='MIT',
    py_modules=['json_numpy'],
    python_requires='>=3.6, <4',
    install_requires=['numpy'],
    project_urls={
        'Bug Reports': 'https://github.com/Crimson-Crow/json-numpy/issues',
        'Source': 'https://github.com/Crimson-Crow/json-numpy',
    },
)

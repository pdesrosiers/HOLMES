import setuptools

setuptools.setup(
    name="holmes",
    version="0.1.0",
    url="https://github.com/pdesrosiers/HOLMES",
    author="Xavier Roy-Pomerleau, Patrick Desrosiers",
    author_email="xavier.roy-pomerleau.1@ulaval.ca",
    description="A Python package to infer higher-order dependencies from presence-absence data",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='simplicial complexes, hypergraphs, log-linear models, ',
    packages=setuptools.find_packages(),
    install_requires=['matplotlib','numpy','numba'],
    python_requires='>=3',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.png'],
        "doc": ['*.html']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',

        'Operating System :: OS Independent'
    ],
)
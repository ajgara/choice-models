import setuptools

setuptools.setup(
    name='python_choice_models',
    version='0.0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'numba', 'scipy', 'ghalton', 'matplotlib'],
    python_requires='>=3.6',
    zip_safe=False
)

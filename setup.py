import setuptools

    
setuptools.setup(
    name="planetary_systems",
    version="0.0.1",
    author="swissai planetary systems team",
    description="A VAE for planetary systems",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "black",
        "pytest"
        ],
    
)
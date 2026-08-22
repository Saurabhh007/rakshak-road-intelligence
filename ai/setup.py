from setuptools import setup, find_packages

setup(
    name="rakshak_ai",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "numpy",
    ],
)

from setuptools import setup, find_packages

setup(
    name="camera-realsense-maniskill-service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["websockets", "numpy", "Pillow"],
)

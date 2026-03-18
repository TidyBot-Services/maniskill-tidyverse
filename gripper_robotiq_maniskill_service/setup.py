from setuptools import setup, find_packages

setup(
    name="gripper-robotiq-maniskill-service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pyzmq", "msgpack", "numpy"],
)

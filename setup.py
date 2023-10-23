from setuptools import setup, find_packages

exec(open("joint/version.py").read())

setup(
    name="joint",
    version=__version__,  # noqa
    packages=find_packages(),
    entry_points={"console_scripts": ["joint = joint.__main__:start_cli"]},
    install_requires=[
        "nptyping",
        "h5py",
        "imagesize",
        "overrides",
        "colorlog",
        "colored_traceback",
        "tqdm"
    ],
)

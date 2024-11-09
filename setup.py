from setuptools import setup, find_packages

exec(open("spr_pick/version.py").read())

setup(
    name="spr_pick",
    version=__version__,  # noqa
    packages=find_packages(),
    entry_points={"console_scripts": ["spr_pick = spr_pick.__main__:start_cli"]},
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

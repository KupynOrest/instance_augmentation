import os
import re
import ast
from setuptools import setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    packages, links = [], []
    for line in lineiter:
        if line and (line.startswith("#") or line.startswith("-")):
            links.append(line)
        else:
            packages.append(line)
    return packages, links


def get_version():
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    main_file = os.path.join(curr_dir, "instance_augmentation", "__init__.py")
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(main_file, "r", encoding="utf8") as f:
        match = _version_re.search(f.read())
        version = match.group("version") if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


if __name__ == "__main__":
    packages, links = parse_requirements("requirements.txt")
    setup(
        name="instance_augmentation",
        version=get_version(),
        author="Orest Kupyn",
        description="Dataset Enhancement with Instance Augmentation Implementation",
        package_dir={"": "."},
        url="https://github.com/KupynOrest/InstanceAugmentation",
        packages=["instance_augmentation"],
        include_package_data=True,
        install_requires=packages,
        dependency_links=links,
    )

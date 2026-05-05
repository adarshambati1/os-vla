from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="os-vla",
    version="0.0.1",
    description="Operational-Space VLA: PaliGemma + LoRA + 24D operational-space action head.",
    author="Adarsh Ambati",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "notebooks", "data", "configs", "docs"]),
    install_requires=requirements,
)

from setuptools import setup, find_packages

# Function to read the contents of your requirements.txt file


def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name='gpt-4v-distribution-shift',
    version='1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    # additional setup configuration...
)

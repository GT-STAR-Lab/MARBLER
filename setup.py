from setuptools import setup, find_packages

setup(name='robotarium_gym',
      version='0.0.1',
      description='Gym Environment for the Robotarium',
      url='',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
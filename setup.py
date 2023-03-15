from setuptools import setup, find_packages

setup(name='robogym',
      version='0.0.1',
      description='Gym Environment for the Robotarium',
      url='',
      author='',
      author_email='rtorbati3@gatech.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
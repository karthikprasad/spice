from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'hmm modules',
  ext_modules = cythonize("_hmmc.pyx"),
)

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from sys import platform as _platform
import os
import numpy as np

#openmp_arg = '-fopenmp'
#if _platform == "win32":
#    openmp_arg = '-openmp'

extensions = [
  Extension(
    'nms_grid', ['nms_grid.pyx'],
    language="c++",
    include_dirs=[np.get_include(), '.','include'],
    extra_compile_args=['-DILOUSESTL','-DIL_STD','-std=c++11','-O3'],
    extra_link_args=['-std=c++11']
  )
] 

setup(
    name = 'nms_grid',
    ext_modules = cythonize(extensions)
)

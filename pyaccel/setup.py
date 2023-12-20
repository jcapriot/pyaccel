import os
from os.path import join, abspath, dirname

base_path = abspath(dirname(__file__))


# Enable line tracing for coverage of cython files conditionally
ext_kwargs = {}
if os.environ.get("TEST_COV", None) is not None:
    ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    import numpy.distutils.system_info as sysinfo
    config = Configuration("py_accel", parent_package, top_path)

    try:
        from Cython.Build import cythonize
        cythonize(join(base_path, "accel_solver.pyx"))
    except ImportError:
        pass

    config.add_extension(
        "accel_solver",
        sources=["accel_solver.c"],
        # include_dirs=get_numpy_include_dirs(),
        extra_compile_args=['-w', '-framework', 'Accelerate'],
        extra_link_args=['-framework', 'Accelerate']
    )

    return config

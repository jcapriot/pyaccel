module_path = 'pyaccel'

py.install_sources(
  '__init__.py',
  subdir: module_path
)

py.extension_module(
    'accel_solver',
    'accel_solver.pyx',
    install: true,
    subdir: module_path,
    dependencies : [dep_py, dep_accel],
)
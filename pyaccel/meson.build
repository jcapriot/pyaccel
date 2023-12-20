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
    link_args : ['-framework', 'Accelerate'],
    # compile_args : ['-framework', 'Accelerate'],
    dependencies : dep_py, #[dep_py, accel_dep],
)
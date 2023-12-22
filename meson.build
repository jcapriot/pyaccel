project(
    'pyaccel',
    'cython', 'c',
    version: '0.0.1',
    license: 'MIT',
    meson_version: '>= 1.1.0',
)

py = import('python').find_installation(pure: false)
dep_py = py.dependency()

cc = meson.get_compiler('c')
dep_accel = dependency('accelerate')

subdir('pyaccel')
project('autopen', 'cpp',
	version : '0.1.0',
	default_options : ['warning_level=3', 'cpp_std=c++14']
)

has_devenv = meson.version().version_compare('>=0.58.0')
if has_devenv
	devenv = environment()
endif

conf_data = configuration_data()
conf_data.set('version', meson.project_version())

gnome = import('gnome')

deps = [
	dependency('gtkmm-3.0'),
	dependency('gtksourceviewmm-3.0'),
	dependency('zlib'),
	dependency('fontconfig'),
    dependency('vulkan'),
	dependency('gdk-x11-3.0', required : false),
    declare_dependency(link_args : ['-Wl,-Bdynamic','-L'+meson.current_source_dir()+'/llama.cpp/bin', '-lllama']),
    declare_dependency(link_args : [meson.current_source_dir()+'/llama.cpp/common/libcommon.a'])
]

src = [
	'main.cpp',
	'mainwindow.cpp',
	'tokentree.cpp'
]

incdir = include_directories('llama.cpp', 'llama.cpp/common/')

if meson.version().version_compare('>=0.56.0')
	executable('autopen', src,
		dependencies: deps,
        include_directories : incdir,
		install : true,
		win_subsystem: 'windows'
	)
else
	executable('autopen', src,
		dependencies: deps,
        include_directories : incdir,
		install : true
	)
endif

if has_devenv
	meson.add_devenv(devenv)
endif

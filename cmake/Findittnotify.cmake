# - Find ittnotify.
# Defines:
# ittnotify_FOUND
# ittnotify_INCLUDE_DIRS
# ittnotify_LIBRARIES

set(root_dir
    ${PROJECT_SOURCE_DIR}/external/ittapi
  )

find_path(ittnotify_INCLUDE_DIRS ittnotify.h
    PATHS ${root_dir}/include
    NO_DEFAULT_PATH)

if (CMAKE_SIZEOF_VOID_P MATCHES "8")
  set(ittnotify_lib_dir 64)
else()
  set(ittnotify_lib_dir 32)
endif()

find_library(ittnotify_LIBRARIES libittnotify
    PATHS ${root_dir}/build_win/${ittnotify_lib_dir}/bin/Release)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ittnotify DEFAULT_MSG ittnotify_LIBRARIES ittnotify_INCLUDE_DIRS)
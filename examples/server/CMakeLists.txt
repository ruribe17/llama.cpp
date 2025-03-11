set(TARGET llama-server)

option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggml-org/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

# Find necessary packages
find_package(CURL REQUIRED)
# find_package(OpenSSL REQUIRED)

if (CURL_FOUND)
    message(STATUS "Found CURL: ${CURL_LIBRARIES}")
else()
    message(FATAL_ERROR "CURL library not found!")
endif()

set(TARGET_SRCS
    server.cpp
    translator.hpp
    translator.cpp
    utils.hpp
    httplib.h
)

set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})

    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )

    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR} ${CURL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE common ${CMAKE_THREAD_LIBS_INIT} ${CURL_LIBRARIES})

if (LLAMA_SERVER_SSL)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

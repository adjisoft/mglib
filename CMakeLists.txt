cmake_minimum_required(VERSION 3.16)

project(mglib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mglib INTERFACE)
target_include_directories(mglib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(mglib INTERFACE cxx_std_17)

option(MGLIB_BUILD_TESTS "Build mglib tests" ON)

if(MGLIB_BUILD_TESTS)
    enable_testing()
    add_executable(mglib_tests tests.cc)
    target_link_libraries(mglib_tests PRIVATE mglib)
    add_test(NAME mglib_tests COMMAND mglib_tests)
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(
        doc_doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

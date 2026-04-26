cmake_minimum_required(VERSION 3.16)
project(morphe-cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED MORPHE_BUILD_TESTS)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(MORPHE_BUILD_TESTS ON)
    else()
        set(MORPHE_BUILD_TESTS OFF)
    endif()
endif()

add_library(morphe STATIC
    src/constraints.cpp
    src/serialization.cpp
    src/validation.cpp
)
target_include_directories(morphe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
)
target_compile_features(morphe PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(morphe PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
elseif(MSVC)
    target_compile_options(morphe PRIVATE /W4)
endif()

add_library(morphe::morphe ALIAS morphe)

if(MORPHE_BUILD_TESTS)
    enable_testing()

    add_executable(morphe_tests
        tests/test_main.cpp
        tests/test_smoke.cpp
        tests/test_types.cpp
        tests/test_primitives.cpp
        tests/test_constraints.cpp
        tests/test_document.cpp
        tests/test_serialization.cpp
        tests/test_validation.cpp
        tests/test_conformance.cpp
    )
    target_link_libraries(morphe_tests PRIVATE morphe)
    target_include_directories(morphe_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )

    # Inject the conformance corpus path. CMake resolves this at configure
    # time; the C++ test reads MORPHE_CORPUS_DIR at compile time.
    target_compile_definitions(morphe_tests PRIVATE
        MORPHE_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/conformance"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(morphe_tests PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    elseif(MSVC)
        target_compile_options(morphe_tests PRIVATE /W4)
        # Force-include <ostream> on MSVC: doctest's single header only
        # forward-declares std::basic_ostream, but recent MSVC standard
        # library headers (VS 2022 17.14+) require the full definition
        # when <string_view> is pulled in transitively. Force-include
        # avoids touching every test source.
        target_compile_options(morphe_tests PRIVATE /FIostream)
    endif()

    add_test(NAME morphe_tests COMMAND morphe_tests)
endif()

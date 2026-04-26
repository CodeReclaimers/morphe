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
    )
    target_link_libraries(morphe_tests PRIVATE morphe)
    target_include_directories(morphe_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(morphe_tests PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    elseif(MSVC)
        target_compile_options(morphe_tests PRIVATE /W4)
    endif()

    add_test(NAME morphe_tests COMMAND morphe_tests)
endif()

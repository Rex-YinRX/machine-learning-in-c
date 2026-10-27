cmake_minimum_required(VERSION 3.25)

project(MachineLearningInC LANGUAGES C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(nn_core STATIC
    src/matrix.c
    src/layer.c
    src/network.c
)

target_include_directories(nn_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(nn_core PUBLIC c_std_11)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(nn_core PUBLIC m)
endif()

if(MSVC)
    target_compile_options(nn_core PRIVATE /W4)
    target_compile_options(nn_core PUBLIC /utf-8)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nn_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(load_model programs/load_model.c)
target_link_libraries(load_model PRIVATE nn_core)

add_executable(continue_train programs/continue_train.c)
target_link_libraries(continue_train PRIVATE nn_core)

add_executable(matrix_demo programs/main.c)
target_link_libraries(matrix_demo PRIVATE nn_core)

add_executable(mnist tools/mnist.c)
target_link_libraries(mnist PRIVATE nn_core)

option(BUILD_DIGIT_APP "Build handwritten digit apps and dataset tools" ON)

if(BUILD_DIGIT_APP)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build raylib examples")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/raylib/CMakeLists.txt")
        add_subdirectory(third_party/raylib)
    else()
        include(FetchContent)

        FetchContent_Declare(raylib
            URL https://github.com/raysan5/raylib/archive/refs/tags/6.0.zip
            URL_HASH SHA256=A6770924FED7B25DE9F6009EC236CD095A99577A3907026DA2BFA21A27AD271A
        )

        FetchContent_MakeAvailable(raylib)
    endif()

    add_executable(digit_app
        programs/digit_app.c
        programs/digit_preprocess.c
    )

    target_link_libraries(digit_app PRIVATE nn_core raylib)

    install(TARGETS digit_app
        RUNTIME DESTINATION .
        COMPONENT digit_app
    )

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/handwritten_10ep.bin"
        DESTINATION model
        RENAME mnist_model.bin
        COMPONENT digit_app
    )

    add_library(handwritten_data STATIC
        tools/handwritten_dataset.c
        programs/digit_preprocess.c
    )
    target_include_directories(handwritten_data PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/tools"
        "${CMAKE_CURRENT_SOURCE_DIR}/programs"
    )
    target_link_libraries(handwritten_data PUBLIC nn_core raylib)

    add_executable(digit_collect tools/digit_collect.c)
    target_link_libraries(digit_collect PRIVATE handwritten_data)

    add_executable(digit_dataset tools/digit_dataset.c)
    target_link_libraries(digit_dataset PRIVATE handwritten_data)

    foreach(handwritten_target handwritten_data digit_collect digit_dataset)
        if(MSVC)
            target_compile_options(${handwritten_target} PRIVATE /W4)
        elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${handwritten_target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endforeach()

endif()

cmake_minimum_required(VERSION 3.20)

project(cis_677_high_performance_computing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HPC_NATIVE "Enable native CPU tuning for optimized builds" OFF)
option(HPC_OPENMP "Enable OpenMP support" OFF)

if(HPC_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

function(add_hpc_entry target_name project_dir entry_dir)
    add_executable(${target_name}
            "${project_dir}/baseline/harness.cpp"
            "${entry_dir}/src/prepare.cpp"
            "${entry_dir}/src/solution.cpp"
    )

    target_include_directories(${target_name} PRIVATE
            "${project_dir}/baseline/include"
    )

    if(HPC_NATIVE)
        target_compile_options(${target_name} PRIVATE
                "$<$<CONFIG:Release>:-march=native>"
                "$<$<CONFIG:RelWithDebInfo>:-march=native>"
        )
    endif()

    if(HPC_OPENMP)
        target_link_libraries(${target_name} PRIVATE
                OpenMP::OpenMP_CXX
        )
    endif()
endfunction()

function(add_hpc_project project_dir)
    get_filename_component(project_name "${project_dir}" NAME)

    add_hpc_entry(
            "${project_name}_baseline"
            "${project_dir}"
            "${project_dir}/baseline"
    )

    file(GLOB iteration_dirs
            LIST_DIRECTORIES true
            CONFIGURE_DEPENDS
            "${project_dir}/iteration-*"
    )

    foreach(entry_dir IN LISTS iteration_dirs)
        if(IS_DIRECTORY "${entry_dir}"
                AND EXISTS "${entry_dir}/src/prepare.cpp"
                AND EXISTS "${entry_dir}/src/solution.cpp")

            get_filename_component(entry_name "${entry_dir}" NAME)
            string(REPLACE "-" "_" entry_name "${entry_name}")

            add_hpc_entry(
                    "${project_name}_${entry_name}"
                    "${project_dir}"
                    "${entry_dir}"
            )
        endif()
    endforeach()
endfunction()

file(GLOB project_dirs
        LIST_DIRECTORIES true
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/p*"
)

foreach(project_dir IN LISTS project_dirs)
    get_filename_component(project_name "${project_dir}" NAME)

    if(IS_DIRECTORY "${project_dir}"
            AND project_name MATCHES "^p[0-9]+$"
            AND EXISTS "${project_dir}/baseline/harness.cpp"
            AND EXISTS "${project_dir}/baseline/include/task.hpp"
            AND EXISTS "${project_dir}/baseline/src/prepare.cpp"
            AND EXISTS "${project_dir}/baseline/src/solution.cpp")

        add_hpc_project("${project_dir}")
    endif()
endforeach()
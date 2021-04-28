cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGUI"          ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use static libigl"  ON)

find_package(LIBIGL REQUIRED)

add_executable(${PROJECT_NAME})

# tetgen
add_subdirectory(src/extern/tetgen)
target_include_directories(${PROJECT_NAME} PRIVATE src/extern/tetgen)

# implot
target_sources(${PROJECT_NAME} PRIVATE
  src/extern/implot/implot.cpp
  src/extern/implot/implot_items.cpp
  )
target_include_directories(${PROJECT_NAME} PRIVATE src/extern/implot)

# TAO
target_include_directories(${PROJECT_NAME} PRIVATE ../petsc/include ../petsc/arch-linux-c-debug/include)
target_link_directories(${PROJECT_NAME} PRIVATE ../petsc/arch-linux-c-debug/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE petsc)

# project files
target_compile_definitions(${PROJECT_NAME} PRIVATE
  EIGEN_INITIALIZE_MATRICES_BY_NAN
  EIGEN_NO_AUTOMATIC_RESIZING)
target_sources(${PROJECT_NAME} PRIVATE
  src/main.cpp
  src/Mesh.cpp
  src/arap_util.cpp
  src/arap_model.cpp
  src/newton_solver.cpp
  src/tao_solver.cpp
  )

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui tet)

add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
target_include_directories(${PROJECT_NAME} PRIVATE external/ tetgen/)


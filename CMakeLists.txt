cmake_minimum_required(VERSION 3.1)
project(example)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGUI"          ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use static libigl"  ON)

find_package(LIBIGL REQUIRED)


# Add your project files
add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
  src/main.cpp
  src/arap.cpp
  src/Mesh.cpp
  )

target_link_libraries(${PROJECT_NAME} igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui tet)

add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
target_include_directories(${PROJECT_NAME} PRIVATE external/ tetgen/)


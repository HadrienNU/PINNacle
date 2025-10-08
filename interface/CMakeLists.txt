cmake_minimum_required(VERSION 3.16)

project(pinn_interface
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options to keep third-party builds minimal
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

include(FetchContent)

# GLFW
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.9 # stable release
)
FetchContent_MakeAvailable(glfw)

# GLM (header-only)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

find_package(OpenGL REQUIRED)

# Récupérer tous les fichiers .cpp dans src/
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  src/*.cpp
)

add_executable(pinn_interface
  ${SOURCES}
)

# Ajouter le dossier include/ aux chemins d'inclusion
target_include_directories(pinn_interface PRIVATE
  include
)

target_link_libraries(pinn_interface PRIVATE
  glfw
  OpenGL::GL
  glm::glm
)

target_compile_definitions(pinn_interface PRIVATE
  GLFW_INCLUDE_NONE # prevent GLFW from including legacy GL headers if a loader is later added
)

target_compile_features(pinn_interface PRIVATE cxx_std_17)

if (MSVC)
  target_compile_options(pinn_interface PRIVATE /W4 /permissive-)
else()
  target_compile_options(pinn_interface PRIVATE -Wall -Wextra -Wpedantic)
endif()

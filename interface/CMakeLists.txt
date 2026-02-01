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

# Find OpenGL
find_package(OpenGL REQUIRED)

# Collect application sources
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
  app/src/*.cpp
  app/src/*.c
)

# Collect external sources (ImGui, GLAD)
set(EXTERNAL_SOURCES
  external/imgui/imgui.cpp
  external/imgui/imgui_draw.cpp
  external/imgui/imgui_tables.cpp
  external/imgui/imgui_widgets.cpp
  external/imgui/backends/imgui_impl_glfw.cpp
  external/imgui/backends/imgui_impl_opengl3.cpp
  external/glad/glad.c
)

add_executable(pinn_interface
  ${APP_SOURCES}
  ${EXTERNAL_SOURCES}
)

target_include_directories(pinn_interface PRIVATE
  app/include
  external
  external/imgui
  external/imgui/backends
  external/glad
  external/stb
  external/KHR
)

target_link_libraries(pinn_interface PRIVATE
  glfw
  glm::glm
)

# Link the proper OpenGL target depending on what's available
if (TARGET OpenGL::GL)
  target_link_libraries(pinn_interface PRIVATE OpenGL::GL)
elseif (TARGET OpenGL::OpenGL)
  target_link_libraries(pinn_interface PRIVATE OpenGL::OpenGL)
else()
  message(FATAL_ERROR "No suitable OpenGL target found from find_package(OpenGL)")
endif()

target_compile_definitions(pinn_interface PRIVATE
  GLFW_INCLUDE_NONE # prevent GLFW from including legacy GL headers if a loader is later added
)

target_compile_features(pinn_interface PRIVATE cxx_std_17)

if (MSVC)
  target_compile_options(pinn_interface PRIVATE /W4 /permissive-)
else()
  target_compile_options(pinn_interface PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Silence OpenGL deprecation warnings on macOS
if (APPLE)
  target_compile_definitions(pinn_interface PRIVATE GL_SILENCE_DEPRECATION)
endif()

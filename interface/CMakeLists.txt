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

# Optional: best-effort auto-install of system deps (currently Linux only)
option(PINN_AUTOINSTALL_DEPS "Attempt to auto-install system OpenGL/X11 deps (Linux only)" OFF)
if (PINN_AUTOINSTALL_DEPS AND UNIX AND NOT APPLE)
  message(STATUS "PINN_AUTOINSTALL_DEPS=ON: attempting to detect distro and install dev packages (may prompt for sudo)")
  # Try a few common package managers in order. This is best-effort and can be skipped if it fails.
  execute_process(COMMAND bash -c "if command -v apt-get >/dev/null 2>&1; then echo apt; elif command -v dnf >/dev/null 2>&1; then echo dnf; elif command -v pacman >/dev/null 2>&1; then echo pacman; elif command -v zypper >/dev/null 2>&1; then echo zypper; else echo none; fi" OUTPUT_VARIABLE PKG_MGR OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (PKG_MGR STREQUAL "apt")
    execute_process(COMMAND bash -c "sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev xorg-dev" RESULT_VARIABLE APT_RES)
  elseif (PKG_MGR STREQUAL "dnf")
    execute_process(COMMAND bash -c "sudo dnf install -y mesa-libGL-devel libX11-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel" RESULT_VARIABLE APT_RES)
  elseif (PKG_MGR STREQUAL "pacman")
    execute_process(COMMAND bash -c "sudo pacman -Syu --noconfirm && sudo pacman -S --noconfirm mesa libx11 libxcursor libxrandr libxinerama libxi" RESULT_VARIABLE APT_RES)
  elseif (PKG_MGR STREQUAL "zypper")
    execute_process(COMMAND bash -c "sudo zypper install -y Mesa-libGL-devel libX11-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel" RESULT_VARIABLE APT_RES)
  else()
    set(APT_RES 0)
  endif()
  if (NOT APT_RES EQUAL 0)
    message(WARNING "Automatic dependency install failed (code=${APT_RES}). Please install OpenGL/X11 dev packages manually.")
  endif()
endif()

# Find OpenGL
find_package(OpenGL REQUIRED)

add_executable(pinn_interface
  src/main.cpp
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

## Prerequisites

On Linux you need basic development tools and OpenGL/X11 dev packages. For example on Ubuntu/Debian:

- build-essential, cmake, git
- libx11-dev, xorg-dev, libglu1-mesa-dev

GLFW and GLM are fetched automatically at configure time via CMake's FetchContent.

## Build and run

- Quick path using Makefile:

```
make -C interface run
```

- Or manually with CMake:

```
cmake -S interface -B interface/build -DCMAKE_BUILD_TYPE=Release
cmake --build interface/build -j
./interface/build/pinn_interface
```

You should see an empty 800x600 window with a dark background.

## Windows

Two common options:

1) Visual Studio (MSVC toolchain):

```
cmake -S interface -B interface/build -G "Visual Studio 17 2022" -A x64
cmake --build interface/build --config Release -j
interface/build/Release/pinn_interface.exe
```

2) MSYS2 MinGW (GCC toolchain):

- Open an MSYS2 MinGW 64-bit shell, then install tools if needed:

```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain cmake git
```

- Then build and run:

```
cmake -S interface -B interface/build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build interface/build -j
./interface/build/pinn_interface.exe
```

Notes:
- GLFW and GLM are fetched automatically by CMake (no separate install needed).
- With MSVC, the executable is in `interface/build/Release/` for the selected configuration.

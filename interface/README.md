## Prerequisites

You'll need basic build tools and OpenGL development headers.

- Linux: OpenGL/X11 dev packages. For Ubuntu/Debian: `build-essential`, `cmake`, `git`, `libx11-dev`, `xorg-dev`, `libglu1-mesa-dev`.
- macOS: Xcode command line tools or full Xcode (provides OpenGL framework). No Homebrew package is strictly required.
- Windows: Either Visual Studio (MSVC + Windows SDK) or MSYS2 MinGW toolchain.

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

## macOS

Ensure Xcode command line tools are installed:

```
xcode-select --install
```

Then build and run:

```
cmake -S interface -B interface/build -DCMAKE_BUILD_TYPE=Release
cmake --build interface/build -j
./interface/build/pinn_interface
```

## Optional auto-install of system deps

You can opt-in to automatic installation of OpenGL/X11 development packages on Linux by passing a flag to CMake. This is best-effort and may prompt for sudo; if it fails, install packages manually and re-run CMake.

Linux example (may prompt for sudo):

```
cmake -S interface -B interface/build -DPINN_AUTOINSTALL_DEPS=ON
cmake --build interface/build -j
```

If auto-install is disabled or fails, install manually and re-run CMake:

- Ubuntu/Debian: `sudo apt-get install -y libgl1-mesa-dev xorg-dev`
- Fedora/RHEL: `sudo dnf install -y mesa-libGL-devel libX11-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel`
- Arch: `sudo pacman -Syu --noconfirm && sudo pacman -S --noconfirm mesa libx11 libxcursor libxrandr libxinerama libxi`
- openSUSE: `sudo zypper install -y Mesa-libGL-devel libX11-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel`
- Windows (MSVC): Install Visual Studio Build Tools and Windows SDK, or on MSYS2: `pacman -S --needed mingw-w64-x86_64-opengl-devel mingw-w64-x86_64-glfw mingw-w64-x86_64-glm`

macOS notes:

- We request an OpenGL 3.3 Core context and use <OpenGL/gl3.h>. On older Macs limited to 3.2, you may need to adjust the requested version.

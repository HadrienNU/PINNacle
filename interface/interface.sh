#!/usr/bin/env bash
# Usage:
#   ./interface/interface.sh [configure|build|run|launch|clean|rebuild]
# Default is 'build'.

set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
CONFIG="Release"

cmd="${1:-build}"

configure() {
  echo "[configure] CMake configure -> ${BUILD_DIR}"
  if [ -n "${CMAKE_ARGS:-}" ]; then
    echo "[configure] Extra CMake args: ${CMAKE_ARGS}"
  fi
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" ${CMAKE_ARGS:-}
}

build() {
  configure
  echo "[build] CMake build"
  cmake --build "$BUILD_DIR" --config "$CONFIG" -j
}

launch_app() {
  local exe="./$BUILD_DIR/pinn_interface"
  if [ -f "./$BUILD_DIR/$CONFIG/pinn_interface" ]; then
    exe="./$BUILD_DIR/$CONFIG/pinn_interface"
  fi
  if [ ! -x "$exe" ]; then
    echo "[launch] Executable not found: $exe"
    echo "[launch] Calling 'run' to build and launch..."
    exec "$0" run
  fi
  echo "[launch] ${exe}"
  "${exe}"
}

clean() {
  echo "[clean] Removing ${BUILD_DIR}"
  rm -rf "$BUILD_DIR"
}

case "$cmd" in
  configure) configure ;;
  build)     build ;;
  run)       build; launch_app ;;
  launch)    launch_app ;;
  rebuild)   clean; build ;;
  clean)     clean ;;
  *) echo "Unknown command: $cmd"; echo "Usage: $0 [configure|build|run|launch|clean|rebuild]"; exit 1;;
esac

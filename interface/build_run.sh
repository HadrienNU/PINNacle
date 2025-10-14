#!/usr/bin/env bash
# Usage:
#   ./interface/build_run.sh [configure|build|run|clean|rebuild]
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

run_app() {
  local exe="./$BUILD_DIR/pinn_interface"
  # If a multi-config generator was used (e.g., Xcode or Ninja Multi-Config), prefer the config subfolder
  if [ -f "./$BUILD_DIR/$CONFIG/pinn_interface" ]; then
    exe="./$BUILD_DIR/$CONFIG/pinn_interface"
  fi
  echo "[run] ${exe}"
  "${exe}"
}

clean() {
  echo "[clean] Removing ${BUILD_DIR}"
  rm -rf "$BUILD_DIR"
}

case "$cmd" in
  configure) configure ;;
  build)     build ;;
  run)       build; run_app ;;
  rebuild)   clean; build ;;
  clean)     clean ;;
  *) echo "Unknown command: $cmd"; echo "Usage: $0 [configure|build|run|clean|rebuild]"; exit 1;;
esac

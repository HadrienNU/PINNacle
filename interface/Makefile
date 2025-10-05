BUILD_DIR := build

.PHONY: all configure build run clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build $(BUILD_DIR) --config Release -j

run: build
	./$(BUILD_DIR)/pinn_interface

clean:
	rm -rf $(BUILD_DIR)

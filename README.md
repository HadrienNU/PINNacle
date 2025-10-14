# PINNacle

## 1) Requirements

- Python 3.9+ recommended
- pip, venv (or conda)
- For the optional C++ interface: CMake (>=3.16), a C++17 compiler, and OpenGL dev headers

### OS-specific notes for the Interface (viewer)

- Linux (Ubuntu/Debian): `sudo apt-get install -y build-essential cmake git libx11-dev xorg-dev libglu1-mesa-dev`
- macOS: Xcode command line tools: `xcode-select --install`
- Windows:
	- Visual Studio 2019/2022 with C++ Desktop workload (MSVC + Windows SDK), or
	- MSYS2 MinGW64 toolchain (GCC): install MSYS2, then in MSYS2 MinGW64 shell run `pacman -S --needed base-devel mingw-w64-x86_64-toolchain cmake git`

GLFW and GLM are fetched automatically by CMake (no separate installation required).

## 2) Setup the Python environment

We provide cross-platform setup scripts that create a virtual environment and install dependencies from `requirements.txt`.

- Linux/macOS:

```bash
./setup.sh
```

- Windows:

```bat
setup.bat
```

These scripts create a virtual environment (`venv`, `venv_unix`, or `venv_win` depending on platform) and install packages from `requirements.txt` (PyTorch, NumPy, SciPy, etc.). DeepXDE is included in this repository (`deepxde/`), so no separate installation is required. If you need a CUDA-enabled PyTorch, install it after running setup following PyTorch’s official instructions for your GPU/driver.

## 3) Run a benchmark

After setting up and activating the virtual environment (see Section 2), you can launch the training benchmark script.

```bash
python benchmark.py 
```

Outputs
- All artifacts are written under `runs/` in a time-stamped experiment folder: `runs/<mm.dd-HH.MM.SS>-<name>/`.
- At the experiment root:
	- `config.json`: run configuration (seed, tasks, args)
	- `script.py.bak`: a copy of the exact Python script that launched the run
	- `result.csv`: aggregated metrics over tasks and repeats (created after training)
- For each task/repeat, a subfolder `<taskIndex>-<repeatIndex>/` contains:
	- `log.txt` and `logerr.txt`: full stdout/stderr logs
	- `loss.txt`: training loss history
	- `errors.txt`: evaluation metrics (if available for the PDE)

Notes
- The default PDE list inside `benchmark.py` can be edited to choose which problems to run. By default, it runs a single example; uncomment or add entries in `pde_list` to batch multiple problems.

## 4) Build and run the C++ interface (optional)

The viewer source lives in `interface/`. Prefer the provided scripts:

- Linux/macOS:

```bash
./interface/build_run.sh run
```

- Windows:

```bat
interface\build_run.bat run
```

This will configure CMake, build the project, and run the executable (`pinn_interface`).

For detailed dependency notes and manual CMake commands, see `interface/README.md`.

## 5) Project structure

- `benchmark.py`: main entry to launch predefined PDE benchmarks
- `trainer.py`: training orchestration and logging to `runs/`
- `src/`: Python source (models, optimizers, PDE definitions)
- `interface/`: C++ OpenGL viewer (CMake project)
- `runs/`: output directory for training logs and CSVs

## Troubleshooting

- Import errors: Ensure you ran `setup.sh`/`setup.bat` and that the virtual environment is active when launching scripts.
- GPU not used: Pass `--device cpu` or ensure CUDA is available and PyTorch sees your GPU (`torch.cuda.is_available()` should be True). Consider installing a CUDA-specific wheel from PyTorch’s website.
- Interface build fails on Linux: Verify `libx11-dev xorg-dev libglu1-mesa-dev` are installed.
- Windows build issues: Ensure either Visual Studio Build Tools (with C++ workload) or MSYS2 MinGW64 is properly installed and in PATH.



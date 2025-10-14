@echo off
REM Usage:
REM   interface\interface.bat [configure|build|run|launch|clean|rebuild]

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "BUILD_DIR=build"
set "CONFIG=Release"
set "CMD=%~1"
if "%CMD%"=="" set "CMD=build"

if /I "%CMD%"=="configure" goto :configure
if /I "%CMD%"=="build" goto :build
if /I "%CMD%"=="run" goto :run
if /I "%CMD%"=="launch" goto :launch
if /I "%CMD%"=="rebuild" goto :rebuild
if /I "%CMD%"=="clean" goto :clean
goto :usage

:configure
echo [configure] CMake configure -> %BUILD_DIR%
REM Optional: support extra CMake args via environment variable CMAKE_ARGS (e.g., -G "MinGW Makefiles" or -A x64)
if not "%CMAKE_ARGS%"=="" (
	echo [configure] Extra CMake args: %CMAKE_ARGS%
)
cmake -S . -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%CONFIG% %CMAKE_ARGS% || goto :cmErr
goto :eof

:build
call "%~f0" configure || goto :cmErr
echo [build] CMake build
cmake --build "%BUILD_DIR%" --config %CONFIG% -j || goto :cmErr
goto :eof

:run
call "%~f0" build || goto :cmErr
call "%~f0" launch
goto :eof

:launch
set "EXE=%BUILD_DIR%\pinn_interface.exe"
if exist "%BUILD_DIR%\%CONFIG%\pinn_interface.exe" set "EXE=%BUILD_DIR%\%CONFIG%\pinn_interface.exe"
if not exist "%EXE%" (
	echo [launch] Executable not found: %EXE%
	echo [launch] Calling 'run' to build and launch...
	call "%~f0" run
	goto :eof
)
echo [launch] %EXE%
%EXE%
goto :eof

:rebuild
call "%~f0" clean
call "%~f0" build
goto :eof

:clean
echo [clean] Removing %BUILD_DIR%
rmdir /s /q "%BUILD_DIR%" 2>nul
goto :eof

:usage
echo Unknown command: %CMD%
echo Usage: interface.bat [configure^|build^|run^|launch^|clean^|rebuild]
exit /b 1

:cmErr
echo CMake failed. Ensure a Windows CMake generator (MSVC or MinGW) is installed and in PATH.
exit /b 1

endlocal

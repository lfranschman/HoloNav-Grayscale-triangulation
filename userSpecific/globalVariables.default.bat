@echo off

set _PROJECT_PATH_WITH_SLASH=%_PROJECT_PATH:\=/%

set PATH= 
set PATH=%PATH%;C:/Windows/system32
set LIBRARY_PATH= 
set CPATH= 
set C_INCLUDE_PATH= 
set CPLUS_INCLUDE_PATH= 
set CMAKE_PREFIX_PATH= 

set _PYTHON_DIR=C:/Program Files/Python38
set PATH=%PATH%;%_PYTHON_DIR%/;%_PYTHON_DIR%/Scripts
set PYTHONPATH=%PYTHONPATH%;%_PROJECT_PATH_WITH_SLASH%python/common
set PYTHONPATH=%PYTHONPATH%;%_PROJECT_PATH_WITH_SLASH%thirdparties/scikit-surgerycalibration

rem set _MSVC_DIR=C:/Program Files/Microsoft Visual Studio/2019/Community
rem set _MSVC_BIN_DIR=%_MSVC_DIR%/VC/Tools/MSVC/14.28.29333/bin/Hostx64/x64
rem set PATH=%PATH%;%_MSVC_BIN_DIR%
rem set VS170COMNTOOLS= 
rem set VS160COMNTOOLS=%_MSVC_DIR%/Common7/Tools/
rem set VS150COMNTOOLS= 
rem set VS140COMNTOOLS= 
rem set VS130COMNTOOLS= 
rem set VS120COMNTOOLS= 
rem set VS110COMNTOOLS= 
rem set VS100COMNTOOLS= 

rem set VCPKG_CMAKE_TOOLCHAIN_FILE=C:/Program Files/vcpkg/scripts/buildsystems/vcpkg.cmake

rem set _CMAKE="C:/Program Files/cmake/bin/cmake.exe"
rem set _COMPILER_NAME="Visual Studio 16 2019"
rem set _COMPILATION_NB_JOBS=8
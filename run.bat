@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
setlocal enabledelayedexpansion

REM ============================================================================
REM Windows Launcher for OpenSR-SRGAN to TriSpecWave-Mamba-UQ Distillation
REM All heavy execution runs remotely on Modal GPU/CPU. Local machine only drives CLI.
REM ============================================================================

set VOLUME_NAME=psc26142-pune-distill
set APP_FILE=modal_app.py

REM 1. Verify Modal CLI is installed
where modal >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Modal CLI is not found on PATH.
    echo Please install Modal by running:
    echo     python -m pip install modal
    exit /b 1
)

REM 2. Verify Modal authentication
call modal profile current >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Modal is not authenticated.
    echo Please run:
    echo     modal setup
    exit /b 1
)

REM Parse first argument
set CMD=%~1
if "%CMD%"=="" goto :help

if /i "%CMD%"=="setup" goto :do_setup
if /i "%CMD%"=="verify" goto :do_verify
if /i "%CMD%"=="download" goto :do_download
if /i "%CMD%"=="prepare" goto :do_prepare
if /i "%CMD%"=="cache" goto :do_cache
if /i "%CMD%"=="distill" goto :do_distill
if /i "%CMD%"=="evaluate" goto :do_evaluate
if /i "%CMD%"=="all" goto :do_all
if /i "%CMD%"=="pull" goto :do_pull
if /i "%CMD%"=="volume" goto :do_volume
if /i "%CMD%"=="logs" goto :do_logs
if /i "%CMD%"=="help" goto :help

echo [ERROR] Unknown command: %CMD%
goto :help


REM ============================================================================
REM SUBCOMMAND: setup
REM ============================================================================
:do_setup
echo ============================================================================
echo [RUNNING] Creating persistent Modal Volume: %VOLUME_NAME%
echo ============================================================================
modal volume create %VOLUME_NAME% --version=2 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Volume %VOLUME_NAME% already exists or created.
)
echo [RUNNING] modal run %APP_FILE%::verify_environment
call modal run %APP_FILE%::verify_environment
if %errorlevel% neq 0 (
    echo [ERROR] Setup verification failed.
    exit /b 1
)
echo [SUCCESS] Setup and remote environment verification complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: verify
REM ============================================================================
:do_verify
echo [RUNNING] modal run %APP_FILE%::verify_environment
call modal run %APP_FILE%::verify_environment
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] modal run %APP_FILE%::run_remote_tests
call modal run %APP_FILE%::run_remote_tests
if %errorlevel% neq 0 exit /b 1

echo [SUCCESS] Environment and shape unit tests passed.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: download
REM ============================================================================
:do_download
echo [RUNNING] modal run %APP_FILE%::download_pune_data
call modal run %APP_FILE%::download_pune_data
if %errorlevel% neq 0 (
    echo [ERROR] Remote download failed.
    exit /b 1
)
echo [SUCCESS] Remote download complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: prepare
REM ============================================================================
:do_prepare
set MAX_PATCHES=%~2
if "%MAX_PATCHES%"=="" set MAX_PATCHES=2000
echo [RUNNING] modal run %APP_FILE%::prepare_pune_patches --max-patches %MAX_PATCHES%
call modal run %APP_FILE%::prepare_pune_patches --max-patches %MAX_PATCHES%
if %errorlevel% neq 0 (
    echo [ERROR] Remote patch preparation failed.
    exit /b 1
)
echo [SUCCESS] Remote patch preparation complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: cache
REM ============================================================================
:do_cache
set DETACH_FLAG=
if /i "%~2"=="--detach" set DETACH_FLAG=--detach
if /i "%~3"=="--detach" set DETACH_FLAG=--detach
set MAX_PATCHES=2000
if not "%~2"=="" if not "%~2"=="--detach" set MAX_PATCHES=%~2

echo [RUNNING] modal run %DETACH_FLAG% %APP_FILE%::cache_teacher_targets --max-patches %MAX_PATCHES%
call modal run %DETACH_FLAG% %APP_FILE%::cache_teacher_targets --max-patches %MAX_PATCHES%
if %errorlevel% neq 0 (
    echo [ERROR] Remote teacher caching failed.
    exit /b 1
)
echo [SUCCESS] Remote teacher caching complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: distill
REM ============================================================================
:do_distill
set DETACH_FLAG=
if /i "%~2"=="--detach" set DETACH_FLAG=--detach
if /i "%~3"=="--detach" set DETACH_FLAG=--detach
set MAX_STEPS=8000
if not "%~2"=="" if not "%~2"=="--detach" set MAX_STEPS=%~2

echo [RUNNING] modal run %DETACH_FLAG% %APP_FILE%::distill_student --max-steps %MAX_STEPS%
call modal run %DETACH_FLAG% %APP_FILE%::distill_student --max-steps %MAX_STEPS%
if %errorlevel% neq 0 (
    echo [ERROR] Remote distillation training failed.
    exit /b 1
)
echo [SUCCESS] Remote distillation complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: evaluate
REM ============================================================================
:do_evaluate
echo [RUNNING] modal run %APP_FILE%::evaluate_distillation
call modal run %APP_FILE%::evaluate_distillation
if %errorlevel% neq 0 (
    echo [ERROR] Remote evaluation failed.
    exit /b 1
)
echo [SUCCESS] Remote evaluation complete.
exit /b 0


REM ============================================================================
REM SUBCOMMAND: all (Two-Phase Execution)
REM ============================================================================
:do_all
echo ============================================================================
echo [PHASE 1] SANITY TEST: Setup, Download, 50 Patches, 10 Steps
echo ============================================================================
call :do_setup
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Running remote shape and unit tests...
call modal run %APP_FILE%::run_remote_tests
if %errorlevel% neq 0 exit /b 1

call :do_download
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Stage 1: Preparing 50 patches...
call modal run %APP_FILE%::prepare_pune_patches --max-patches 50
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Stage 1: Caching 50 teacher targets on NVIDIA A10G...
call modal run %APP_FILE%::cache_teacher_targets --max-patches 50
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Stage 1: Distillation 10-step sanity check...
call modal run %APP_FILE%::distill_student --max-steps 10 --num-patches 50
if %errorlevel% neq 0 exit /b 1

echo ============================================================================
echo [PHASE 1 PASSED] 50-patch, 10-step sanity check succeeded!
echo ============================================================================
echo Ready to launch Phase 2: Full 500-patch, 3,000-step distillation.
set /p CONFIRM="Proceed with Phase 2 full training? (Y/N): "
if /i not "!CONFIRM!"=="Y" (
    echo [INFO] Stopped before Phase 2. Run 'run.bat distill' whenever ready.
    exit /b 0
)

echo ============================================================================
echo [PHASE 2] FULL DISTILLATION: 500 Patches, 3,000 Steps
echo ============================================================================
echo [RUNNING] Preparing 500 patches...
call modal run %APP_FILE%::prepare_pune_patches --max-patches 500
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Caching 500 teacher targets...
call modal run %APP_FILE%::cache_teacher_targets --max-patches 500
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Distillation training (3,000 steps on NVIDIA A10G)...
call modal run %APP_FILE%::distill_student --max-steps 3000
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Evaluating distillation against bicubic baseline...
call modal run %APP_FILE%::evaluate_distillation
if %errorlevel% neq 0 exit /b 1

echo [RUNNING] Pulling checkpoints and visual outputs...
call :do_pull
if %errorlevel% neq 0 exit /b 1

echo ============================================================================
echo [SUCCESS] Full distillation pipeline completed successfully!
echo ============================================================================
exit /b 0


REM ============================================================================
REM SUBCOMMAND: pull
REM ============================================================================
:do_pull
if not exist "outputs" mkdir outputs
if not exist "outputs\checkpoints" mkdir outputs\checkpoints

echo [RUNNING] Downloading checkpoints and reports from Modal Volume: %VOLUME_NAME%
modal volume get %VOLUME_NAME% /checkpoints/best.pt ./outputs/best.pt 2>nul
if %errorlevel% neq 0 (
    modal volume get %VOLUME_NAME% checkpoints/best.pt ./outputs/best.pt
)

modal volume get %VOLUME_NAME% /metrics/metrics.json ./outputs/metrics.json 2>nul
if %errorlevel% neq 0 (
    modal volume get %VOLUME_NAME% metrics/metrics.json ./outputs/metrics.json
)

modal volume get %VOLUME_NAME% /outputs/pune_comparison.png ./outputs/pune_comparison.png 2>nul
if %errorlevel% neq 0 (
    modal volume get %VOLUME_NAME% outputs/pune_comparison.png ./outputs/pune_comparison.png
)

modal volume get %VOLUME_NAME% /outputs/pune_uncertainty.png ./outputs/pune_uncertainty.png 2>nul
if %errorlevel% neq 0 (
    modal volume get %VOLUME_NAME% outputs/pune_uncertainty.png ./outputs/pune_uncertainty.png
)

modal volume get %VOLUME_NAME% /outputs/pune_student_rgbn.png ./outputs/pune_student_rgbn.png 2>nul
if %errorlevel% neq 0 (
    modal volume get %VOLUME_NAME% outputs/pune_student_rgbn.png ./outputs/pune_student_rgbn.png
)

echo [SUCCESS] Artifacts downloaded to ./outputs/ directory:
dir outputs
exit /b 0


REM ============================================================================
REM SUBCOMMAND: volume
REM ============================================================================
:do_volume
echo [RUNNING] modal run %APP_FILE%::list_volume_outputs
call modal run %APP_FILE%::list_volume_outputs
exit /b 0


REM ============================================================================
REM SUBCOMMAND: logs
REM ============================================================================
:do_logs
echo [RUNNING] modal app list
call modal app list
echo.
echo To view logs for a running app, execute:
echo   modal app logs [APP_ID]
exit /b 0


REM ============================================================================
REM HELP
REM ============================================================================
:help
echo.
echo ============================================================================
echo Sentinel-2 L2A Distillation Launcher (Modal GPU Remote Runner)
echo ============================================================================
echo Usage: run.bat [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   setup                 Create Modal Volume and verify remote GPU environment
echo   verify                Run remote environment check and automated shape tests
echo   download              Download Pune Sentinel-2 L2A scenes remotely
echo   prepare [NUM]         Extract 64x64 patches (default: 500)
echo   cache [NUM] [--detach] Cache OpenSR-SRGAN RGB-NIR teacher targets on GPU
echo   distill [STEPS] [--detach] Run student distillation on NVIDIA A10G (default: 3000 steps)
echo   evaluate              Evaluate student against bicubic baseline and teacher
echo   all                   Execute two-phase pipeline (50-patch sanity test, then full 3000 steps)
echo   pull                  Download best checkpoint, metrics, and plots to ./outputs/
echo   volume                Display file summary on persistent Modal Volume
echo   logs                  List active Modal apps and tail logs
echo.
echo Examples:
echo   run.bat setup
echo   run.bat distill 3000 --detach
echo   run.bat pull
echo ============================================================================
exit /b 0

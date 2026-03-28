@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Train All Models for Counting Task (Fixed Version)
:: =============================================================================

:: Default Parameters
set "EPOCHS=100"
set "PATIENCE=15"

:: Simple Positional Argument Parsing: if %1 exists, it's Epochs. If %2 exists, it's Patience.
if not "%~1"=="" set "EPOCHS=%~1"
if not "%~2"=="" set "PATIENCE=%~2"

:: Configuration
set "SCRIPT_DIR=%~dp0"
set "WEIGHTS_DIR=%SCRIPT_DIR%weights"
cd /d "%SCRIPT_DIR%"

:: Check Python availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9 or higher.
    exit /b 1
)

echo =================================================================
echo  Train All Models for COUNTING Task
echo =================================================================
echo  Configuration:
echo    Task:      counting
echo    Epochs:    %EPOCHS%
echo    Patience:  %PATIENCE%
echo    Weights:   %WEIGHTS_DIR%
echo.

:: Create weight directories
echo Setting Up Directory Structure...
set "MODELS=jepa jepa_acoustic lewm jepa_sigreg"
set "DATASETS=easy medium hard extreme"

for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        if not exist "%WEIGHTS_DIR%\%%M_%%D" (
            mkdir "%WEIGHTS_DIR%\%%M_%%D" >nul 2>&1
        )
    )
)
echo [SUCCESS] Created all weight directories.
echo.

echo =================================================================
echo  Starting Training Pipeline
echo =================================================================

:: We use a call block to avoid "unexpected at this time" errors within the FOR loop
for %%D in (%DATASETS%) do (
    call :TRAIN_LOOP %%D
)

goto :COMPLETE

:TRAIN_LOOP
set "DS=%1"

:: 1. Train JEPA
echo [PROCESS] Training JEPA on %DS%...
python train.py jepa --dataset %DS% --epochs %EPOCHS% --patience %PATIENCE% --weights-dir "%WEIGHTS_DIR%\jepa_%DS%" --task counting --with-aug
if !errorlevel! equ 0 (echo [SUCCESS] JEPA %DS% complete) else (echo [FAILED] JEPA %DS%)

:: 2. JEPA Acoustic-Only
echo [INFO] JEPA Acoustic-Only %DS%: skipping (shares JEPA weights)

:: 3. Train LeWM
echo [PROCESS] Training LEWM on %DS%...
python train.py lewm --dataset %DS% --epochs %EPOCHS% --patience %PATIENCE% --weights-dir "%WEIGHTS_DIR%\lewm_%DS%" --task counting
if !errorlevel! equ 0 (echo [SUCCESS] LEWM %DS% complete) else (echo [FAILED] LEWM %DS%)

:: 4. Train JEPA+SigReg
echo [PROCESS] Training JEPA_SIGREG on %DS%...
python train_jepa_sigreg.py --dataset %DS% --epochs %EPOCHS% --patience %PATIENCE% --weights-dir "%WEIGHTS_DIR%\jepa_sigreg_%DS%" --task counting --with-aug
if !errorlevel! equ 0 (echo [SUCCESS] JEPA_SIGREG %DS% complete) else (echo [FAILED] JEPA_SIGREG %DS%)

echo -----------------------------------------------------------------
goto :eof

:COMPLETE
echo.
echo =================================================================
echo  Training Pipeline Complete!
echo =================================================================
echo Next steps:
echo   1. Evaluate models in simulation
echo   2. Run evaluation script: python evaluate_counting_all.py
echo   3. Generate results table: python generate_table.py --task counting
echo.
pause
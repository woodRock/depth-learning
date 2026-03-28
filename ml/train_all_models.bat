@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Train All Models Script (CMD Batch Version)
:: =============================================================================
:: Default Values
set EPOCHS=100
set PATIENCE=15

:: Check for command line arguments
if not "%~1"=="" set EPOCHS=%~1
if not "%~2"=="" set PATIENCE=%~2

set SCRIPT_DIR=%~dp0
set WEIGHTS_DIR=%SCRIPT_DIR%weights

echo =================================================================
echo  🚀 Training All Models for Depth Learning Benchmark
echo =================================================================
echo Configuration:
echo   - Epochs:   %EPOCHS%
echo   - Patience: %PATIENCE%
echo   - Weights:  %WEIGHTS_DIR%
echo.

:: 1. Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+.
    exit /b 1
)

:: 2. Create Directory Structure
echo --- Setting Up Directory Structure ---
for %%D in (easy medium hard extreme) do (
    for %%M in (jepa lewm) do (
        if not exist "%WEIGHTS_DIR%\%%M_%%D" (
            mkdir "%WEIGHTS_DIR%\%%M_%%D"
            echo [OK] Created %%M_%%D
        )
    )
)

:: 3. Run Training Loop
echo.
echo =================================================================
echo  📚 Starting Training Pipeline
echo =================================================================

for %%M in (jepa lewm) do (
    echo.
    echo --- %%M Models ---
    for %%D in (easy medium hard extreme) do (
        echo.
        echo [STARTING] %%M on %%D...
        
        :: Run the python command
        python train.py %%M ^
            --dataset %%D ^
            --epochs %EPOCHS% ^
            --patience %PATIENCE% ^
            --weights-dir "%WEIGHTS_DIR%\%%M_%%D"
            
        if !errorlevel! equ 0 (
            echo [SUCCESS] %%M %%D completed.
        ) else (
            echo [FAILED] %%M %%D failed with exit code !errorlevel!.
        )
    )
)

echo.
echo =================================================================
echo  ✅ Training Pipeline Complete
echo =================================================================
pause
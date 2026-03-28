# =============================================================================
# Train All Models Script (PowerShell)
# =============================================================================
# This script trains JEPA and LeWM models on all difficulty datasets
# and saves weights in the correct directories for the Bevy simulation.
#
# Usage:
#   .\train_all_models.ps1 -Epochs 100 -Patience 15
#   .\train_all_models.ps1  # Uses defaults (100 epochs, 15 patience)
#
# Parameters:
#   Epochs   - Number of training epochs (default: 100)
#   Patience - Early stopping patience (default: 15)
#
# Example:
#   .\train_all_models.ps1 -Epochs 100 -Patience 15
# =============================================================================

param(
    [Parameter(Mandatory=$false)]
    [int]$Epochs = 100,
    
    [Parameter(Mandatory=$false)]
    [int]$Patience = 15
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WeightsDir = Join-Path $ScriptDir "weights"

# Helper functions
function Print-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Blue
    Write-Host $Text -ForegroundColor Blue
    Write-Host "=================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Print-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "--- $Text ---" -ForegroundColor Yellow
    Write-Host ""
}

function Print-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Print-Error {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

# Check Python is available
try {
    $pythonCmd = Get-Command python3 -ErrorAction Stop
} catch {
    try {
        $pythonCmd = Get-Command python -ErrorAction Stop
    } catch {
        Print-Error "Python not found. Please install Python 3.9 or higher."
        exit 1
    }
}

$PythonExe = $pythonCmd.Name

# Navigate to script directory
Set-Location $ScriptDir

Print-Header "🚀 Training All Models for Depth Learning Benchmark"
Write-Host "Configuration:"
Write-Host "  Epochs:   $Epochs"
Write-Host "  Patience: $Patience"
Write-Host "  Weights:  $WeightsDir"
Write-Host ""

# Create weights directory structure
Print-Section "Setting Up Directory Structure"
$datasets = @("easy", "medium", "hard", "extreme")
foreach ($dataset in $datasets) {
    $jepaDir = Join-Path $WeightsDir "jepa_$dataset"
    $lewmDir = Join-Path $WeightsDir "lewm_$dataset"
    
    if (!(Test-Path $jepaDir)) {
        New-Item -ItemType Directory -Path $jepaDir | Out-Null
    }
    if (!(Test-Path $lewmDir)) {
        New-Item -ItemType Directory -Path $lewmDir | Out-Null
    }
    Print-Success "Created directories for $dataset"
}

# Track training results
$TrainingResults = @{}

# Training function
function Train-Model {
    param(
        [string]$ModelType,
        [string]$Dataset
    )
    
    $WeightsPath = Join-Path $WeightsDir "${ModelType}_${Dataset}"
    
    Print-Section "Training $($ModelType.ToUpper()) on $($Dataset.ToUpper()) Dataset"
    Write-Host "Weights will be saved to: $WeightsPath"
    
    $arguments = @(
        "train.py",
        $ModelType,
        "--dataset", $Dataset,
        "--epochs", $Epochs.ToString(),
        "--patience", $Patience.ToString(),
        "--weights-dir", $WeightsPath
    )
    
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $PythonExe
    $processInfo.Arguments = $arguments -join " "
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.WorkingDirectory = $ScriptDir
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    
    # Start process and capture output
    $process.Start() | Out-Null
    $output = $process.StandardOutput.ReadToEnd()
    $errorOutput = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    
    if ($process.ExitCode -eq 0) {
        Print-Success "$($ModelType.ToUpper()) $($Dataset.ToUpper()) training complete!"
        $TrainingResults["${ModelType}_${Dataset}"] = "✓ Success"
    } else {
        Print-Error "$($ModelType.ToUpper()) $($Dataset.ToUpper()) training failed!"
        Write-Host $errorOutput -ForegroundColor Red
        $TrainingResults["${ModelType}_${Dataset}"] = "✗ Failed"
    }
}

# Train all models
Print-Header "📚 Starting Training Pipeline"

# JEPA Models (Multi-modal with visual teacher)
Print-Section "JEPA Models (Multi-modal)"
foreach ($dataset in $datasets) {
    Train-Model -ModelType "jepa" -Dataset $dataset
}

# LeWM Models (Acoustic-only)
Print-Section "LeWM Models (Acoustic-only)"
foreach ($dataset in $datasets) {
    Train-Model -ModelType "lewm" -Dataset $dataset
}

# Print summary
Print-Header "📊 Training Summary"
Write-Host ""
Write-Host ("{0,-20} | {1,-15}" -f "Model", "Status")
Write-Host ("{0,-20}-+-{1,-15}" -f ("-"*20), ("-"*15))

foreach ($key in $TrainingResults.Keys) {
    Write-Host ("{0,-20} | {1,-15}" -f $key, $TrainingResults[$key])
}

Write-Host ""
Print-Header "✅ Training Pipeline Complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start the Python server:  python serve.py"
Write-Host "  2. Run the Bevy simulation:  cargo run --release"
Write-Host "  3. Evaluate models in simulation using the Test Evaluation UI"
Write-Host ""
Write-Host "Weights are saved in: $WeightsDir"
Write-Host "  - jepa_easy/, jepa_medium/, jepa_hard/, jepa_extreme/"
Write-Host "  - lewm_easy/, lewm_medium/, lewm_hard/, lewm_extreme/"
Write-Host ""

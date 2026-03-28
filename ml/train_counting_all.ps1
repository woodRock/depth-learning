# =============================================================================
# Train All Models for Counting Task (PowerShell)
# =============================================================================
# This script trains all model architectures on the counting task
# across all difficulty datasets.
#
# Models:
#   - JEPA (Multi-modal with visual teacher)
#   - JEPA Acoustic-Only
#   - LeWM (Acoustic-only)
#   - JEPA+SigReg (Multi-modal with Gaussian regularization)
#
# Usage:
#   .\train_counting_all.ps1 -Epochs 100 -Patience 15
#   .\train_counting_all.ps1  # Uses defaults (100 epochs, 15 patience)
#
# Parameters:
#   Epochs   - Number of training epochs (default: 100)
#   Patience - Early stopping patience (default: 15)
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

Set-Location $ScriptDir

Print-Header "🔢 Training All Models for COUNTING Task"
Write-Host "Configuration:"
Write-Host "  Task:     counting"
Write-Host "  Epochs:   $Epochs"
Write-Host "  Patience: $Patience"
Write-Host "  Weights:  $WeightsDir"
Write-Host ""

# Create weights directory structure
Print-Section "Setting Up Directory Structure"
$models = @("jepa", "jepa_acoustic", "lewm", "jepa_sigreg")
$datasets = @("easy", "medium", "hard", "extreme")

foreach ($model in $models) {
    foreach ($dataset in $datasets) {
        $dir = Join-Path $WeightsDir "${model}_${dataset}"
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
        }
    }
}
Print-Success "Created all weight directories"

# Track training results
$TrainingResults = @{}

# Training function
function Train-Model {
    param(
        [string]$ModelType,
        [string]$Dataset
    )
    
    $WeightsPath = Join-Path $WeightsDir "${ModelType}_${Dataset}"
    
    Print-Section "Training $($ModelType.ToUpper()) on $($Dataset.ToUpper()) Dataset (Counting Task)"
    Write-Host "Weights will be saved to: $WeightsPath"
    
    $arguments = @()
    
    switch ($ModelType) {
        "jepa" {
            $arguments = @(
                "train.py", "jepa",
                "--dataset", $Dataset,
                "--epochs", $Epochs.ToString(),
                "--patience", $Patience.ToString(),
                "--weights-dir", $WeightsPath,
                "--task", "counting",
                "--with-aug"
            )
        }
        "jepa_acoustic" {
            # No training needed - shares JEPA weights
            $TrainingResults["${ModelType}_${Dataset}"] = "✓ (shares JEPA weights)"
            return
        }
        "lewm" {
            $arguments = @(
                "train.py", "lewm",
                "--dataset", $Dataset,
                "--epochs", $Epochs.ToString(),
                "--patience", $Patience.ToString(),
                "--weights-dir", $WeightsPath,
                "--task", "counting"
            )
        }
        "jepa_sigreg" {
            $arguments = @(
                "train_jepa_sigreg.py",
                "--dataset", $Dataset,
                "--epochs", $Epochs.ToString(),
                "--patience", $Patience.ToString(),
                "--weights-dir", $WeightsPath,
                "--task", "counting",
                "--with-aug"
            )
        }
    }
    
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $PythonExe
    $processInfo.Arguments = $arguments -join " "
    $processInfo.UseShellExecute = $false
    $processInfo.WorkingDirectory = $ScriptDir
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start() | Out-Null
    $process.WaitForExit()
    
    if ($process.ExitCode -eq 0) {
        Print-Success "$($ModelType.ToUpper()) $($Dataset.ToUpper()) training complete!"
        $TrainingResults["${ModelType}_${Dataset}"] = "✓ Success"
    } else {
        Print-Error "$($ModelType.ToUpper()) $($Dataset.ToUpper()) training failed!"
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

# JEPA Acoustic-Only (same weights, different evaluation)
Print-Section "JEPA Acoustic-Only (uses same weights as JEPA)"
foreach ($dataset in $datasets) {
    $TrainingResults["jepa_acoustic_${dataset}"] = "✓ (shares JEPA weights)"
}

# LeWM Models (Acoustic-only)
Print-Section "LeWM Models (Acoustic-only)"
foreach ($dataset in $datasets) {
    Train-Model -ModelType "lewm" -Dataset $dataset
}

# JEPA+SigReg Models (Multi-modal with SigReg)
Print-Section "JEPA+SigReg Models (Multi-modal + Gaussian Reg.)"
foreach ($dataset in $datasets) {
    Train-Model -ModelType "jepa_sigreg" -Dataset $dataset
}

# Print summary
Print-Header "📊 Training Summary"
Write-Host ""
Write-Host ("{0,-25} | {1,-15}" -f "Model", "Status")
Write-Host ("{0,-25}-+-{1,-15}" -f ("-"*25), ("-"*15))

foreach ($key in $TrainingResults.Keys) {
    Write-Host ("{0,-25} | {1,-15}" -f $key, $TrainingResults[$key])
}

Write-Host ""
Print-Header "✅ Training Pipeline Complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Evaluate models in simulation OR"
Write-Host "  2. Run evaluation script: python evaluate_counting_all.py"
Write-Host "  3. Generate results table: python generate_table.py --task counting"
Write-Host ""
Write-Host "Weights saved in: $WeightsDir"
Write-Host ""

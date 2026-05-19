# scripts/setup_env_windows.ps1
# Create a reproducible Python virtual environment for the GA project on Windows.

param(
    [string]$PythonVersion = "3.11",
    [string]$TcPythonWheel = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"

Write-Host "Project root: $ProjectRoot"
Write-Host "Requested Python version: $PythonVersion"

Write-Host "Checking Python..."
py -$PythonVersion -c "import sys; print(sys.executable); print(sys.version)"

if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion is not available via py launcher. Run scripts\inspect_server_env.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($TcPythonWheel)) {
    Write-Host "No TC-Python wheel path provided. Searching common locations..."

    $CandidateRoots = @(
        "$env:USERPROFILE\Documents\Thermo-Calc",
        "C:\Program Files\Thermo-Calc",
        "C:\Program Files (x86)\Thermo-Calc",
        "D:\Thermo-Calc",
        "D:\"
    )

    $found = @()

    foreach ($root in $CandidateRoots) {
        if (Test-Path $root) {
            $found += Get-ChildItem -Path $root -Recurse -Filter "TC_Python-*.whl" -ErrorAction SilentlyContinue
        }
    }

    if ($found.Count -eq 0) {
        throw "No TC_Python-*.whl found. Provide -TcPythonWheel explicitly."
    }

    $TcPythonWheel = ($found | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

if (-not (Test-Path $TcPythonWheel)) {
    throw "TC-Python wheel not found: $TcPythonWheel"
}

Write-Host "Using TC-Python wheel: $TcPythonWheel"

if (Test-Path $VenvDir) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item $VenvDir -Recurse -Force
}

Write-Host "Creating virtual environment..."
py -$PythonVersion -m venv $VenvDir

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Python executable: $PythonExe"

Write-Host "Upgrading pip..."
& $PythonExe -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
& $PythonExe -m pip install -r requirements.txt

Write-Host "Installing TC-Python wheel..."
& $PythonExe -m pip install $TcPythonWheel

Write-Host "Environment ready."
& $PythonExe --version

Write-Host "Running environment check..."
& $PythonExe .\scripts\check_environment.py
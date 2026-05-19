# scripts/inspect_server_env.ps1
# Inspect available Python installations and Thermo-Calc/TC-Python paths on Windows.

$ErrorActionPreference = "Continue"

Write-Host "============================================================"
Write-Host "Python launcher versions"
Write-Host "============================================================"
py -0p

Write-Host ""
Write-Host "============================================================"
Write-Host "Default Python"
Write-Host "============================================================"
py --version
py -c "import sys; print(sys.executable); print(sys.version)"

Write-Host ""
Write-Host "============================================================"
Write-Host "Check common Python versions"
Write-Host "============================================================"

$Versions = @("3.14", "3.13", "3.12", "3.11", "3.10", "3.9")

foreach ($v in $Versions) {
    Write-Host ""
    Write-Host "---- Python $v ----"
    py -$v -c "import sys; print(sys.executable); print(sys.version)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        py -$v -c "import importlib.util; mods=['numpy','pandas','sklearn','joblib','yaml','tqdm','deap','tc_python']; [print(m, 'OK' if importlib.util.find_spec(m) else 'MISSING') for m in mods]"
    }
    else {
        Write-Host "Python $v not available through py launcher."
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Search for TC-Python wheel"
Write-Host "============================================================"

$CandidateRoots = @(
    "$env:USERPROFILE\Documents\Thermo-Calc",
    "C:\Program Files\Thermo-Calc",
    "C:\Program Files (x86)\Thermo-Calc",
    "D:\Thermo-Calc",
    "D:\"
)

foreach ($root in $CandidateRoots) {
    if (Test-Path $root) {
        Write-Host "Searching in $root"
        Get-ChildItem -Path $root -Recurse -Filter "TC_Python-*.whl" -ErrorAction SilentlyContinue |
            Select-Object FullName, Length, LastWriteTime
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Done"
Write-Host "============================================================"
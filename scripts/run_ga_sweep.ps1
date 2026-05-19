# scripts/run_ga_sweep.ps1
# Sequential GA sweep over GA mode, population size, generations and random seed.

# powershell -ExecutionPolicy Bypass -File .\scripts\run_ga_sweep.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$BaseConfig = Join-Path $ProjectRoot "config\config.yaml"
$TempConfigDir = Join-Path $ProjectRoot "outputs\temp_configs"
$CacheDir = Join-Path $ProjectRoot "outputs\cache"

New-Item -ItemType Directory -Force -Path $TempConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

# ------------------------------------------------------------
# Define sweep here
# ------------------------------------------------------------

$Modes = @(
    "bulk_dominant",
    "surface_enhanced"
)

$PopGenPairs = @(
    @{ pop = 10; gen = 5 }
    #@{ pop = 60; gen = 20 }
    #@{ pop = 80; gen = 30 }
)

# cambiar seed para prueba 
$Seeds = @(1, 2, 3)

# ------------------------------------------------------------
# Run sweep
# ------------------------------------------------------------

foreach ($mode in $Modes) {
    foreach ($pg in $PopGenPairs) {
        foreach ($seed in $Seeds) {

            $pop = $pg.pop
            $gen = $pg.gen

            $RunLabel = "${mode}_pop${pop}_gen${gen}_seed${seed}"
            $TempConfig = Join-Path $TempConfigDir "config_${RunLabel}.yaml"

            Write-Host ""
            Write-Host "============================================================"
            Write-Host "Running GA: mode=$mode population=$pop generations=$gen seed=$seed"
            Write-Host "Temp config: $TempConfig"
            Write-Host "============================================================"

            # Create temporary YAML by modifying the base config.
            $pythonCode = @"
from pathlib import Path
import yaml

base_path = Path(r"$BaseConfig")
out_path = Path(r"$TempConfig")

with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["ga"]["mode"] = "$mode"
cfg["ga"]["population_size"] = int($pop)
cfg["ga"]["n_generations"] = int($gen)
cfg["ga"]["random_seed"] = int($seed)

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(out_path)
"@

            $pythonCode | py -3.11

            Write-Host "Removing cache files..."
            Remove-Item (Join-Path $CacheDir "ga_cache_*.pkl") -ErrorAction SilentlyContinue

            $Elapsed = Measure-Command {
                py -3.11 .\scripts\run_ga.py --config $TempConfig

                if ($LASTEXITCODE -ne 0) {
                    throw "run_ga.py failed with exit code $LASTEXITCODE for $RunLabel"
                }
            }

            Write-Host "Completed: $RunLabel"
            Write-Host ("Elapsed: {0:hh\:mm\:ss}" -f $Elapsed)

            # Short pause to allow TC-Python / Java resources to close cleanly.
            Start-Sleep -Seconds 5
        }
    }
}

Write-Host ""
Write-Host "SWEEP COMPLETE"
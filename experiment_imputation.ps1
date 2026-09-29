# Learning-rate schedule sweep (ADR 0003), tier 1: cheap datasets, every non-legacy schedule.
#
#   .\experiment.ps1            # run the sweep
#   .\experiment.ps1 -DryRun    # only print what would run
#
# Epoch counts cannot be set from the CLI, so the script writes a hyperparameter JSON
# per dataset with EPOCHS_PRE = $PretrainEpochs, runs every schedule through
# --lr_scheduler, and then restores the original file (or deletes the one it created).

[CmdletBinding()]
param(
    [string[]]$Datasets = @("vehicle_00nan", "credit-g_00nan"),
    [string[]]$Schedulers = @("cosine", "warmup_cosine", "constant", "plateau"),
    [int]$PretrainEpochs = 200,
    [int]$FinetuneEpochs = 150,
    [int]$CvFolds = 3,
    [int]$Seed = 420,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-HyperparamsPath([string]$dataset) {
    $base = $dataset.Split("_")[0]
    # Nested Join-Path: the three-argument form needs PowerShell 7, this runs on 5.1 too.
    return Join-Path (Join-Path "datasets/hiperparams" $base) "$dataset.json"
}

# Everything except EPOCHS_PRE/EPOCH_FINE matches the defaults in src/training/types.py.
function New-HyperparamsJson() {
    return [ordered]@{
        DIM = 128; HIDDEN_DIM = 16; HEADS = 16; LAYERS = 2; DIM_FEED = 32; DROPOUT = 0.2
        EPOCHS_PRE = $PretrainEpochs; BATCH = 256
        LR_PRE = 0.00034; WEIGHT_DECAY_PRE = 0.005; PROB_MASCARA = 0.5
        EPOCH_FINE = $FinetuneEpochs; LR_FINE = 0.001; WEIGHT_DECAY_FINE = 0.0019
        LABELS = 4
    } | ConvertTo-Json
}

# Back up whatever hyperparameter files exist, then write the sweep configuration.
$backups = @{}
foreach ($dataset in $Datasets) {
    $path = Get-HyperparamsPath $dataset
    $backups[$path] = if (Test-Path $path) { [System.IO.File]::ReadAllText((Resolve-Path $path)) } else { $null }
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
        # .NET writes UTF-8 without a BOM on every PowerShell version; Set-Content -Encoding utf8
        # adds a BOM on 5.1, which the trainer's JSON loader rejects.
        [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot $path), (New-HyperparamsJson))
    }
    Write-Host "[setup] $path -> EPOCHS_PRE=$PretrainEpochs EPOCH_FINE=$FinetuneEpochs"
}

$results = @()
$sweepStart = Get-Date
try {
    foreach ($dataset in $Datasets) {
        foreach ($scheduler in $Schedulers) {
            $uvArgs = @("run", "--python", "3.10", "main.py", "--task", "imputation",
                      "--dataset_name", $dataset, "--cv_folds", $CvFolds,
                      "--seed", $Seed, "--lr_scheduler", $scheduler)
            Write-Host ""
            Write-Host "=== $dataset | $scheduler ===" -ForegroundColor Cyan
            Write-Host "uv $($uvArgs -join ' ')"
            if ($DryRun) { continue }

            $start = Get-Date
            & uv @uvArgs
            $status = if ($LASTEXITCODE -eq 0) { "OK" } else { "FAILED ($LASTEXITCODE)" }
            $elapsed = (Get-Date) - $start
            $results += [pscustomobject]@{
                Dataset = $dataset; Scheduler = $scheduler; Status = $status
                Minutes = [math]::Round($elapsed.TotalMinutes, 1)
            }
            Write-Host "[$status] $dataset / $scheduler in $([math]::Round($elapsed.TotalMinutes, 1)) min"
        }
    }
}
finally {
    # Put the hyperparameter files back exactly as they were, even on Ctrl+C.
    if (-not $DryRun) {
        foreach ($path in $backups.Keys) {
            if ($null -eq $backups[$path]) {
                Remove-Item $path -Force -ErrorAction SilentlyContinue
                Write-Host "[cleanup] removed $path"
            } else {
                [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot $path), $backups[$path])
                Write-Host "[cleanup] restored $path"
            }
        }
    }
}

if ($results.Count -gt 0) {
    Write-Host ""
    Write-Host "SWEEP SUMMARY (total $([math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)) min)" -ForegroundColor Green
    $results | Format-Table -AutoSize
    Write-Host "Compare in MLflow: tags.run_role = 'parent' and tags.missingness_percent = '0', grouped by tags.lr_scheduler"
    Write-Host "Metric: cv/test/f1_macro/mean with cv/test/f1_macro/ci95_lower and ci95_upper"
}

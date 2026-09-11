# Reduced Optuna search for the imputation task (ADR 0005): one study per (dataset,
# variant) pair, each promoting its winner to
# datasets/hiperparams/<base>/<base>_<variant>.imputation.json; then, with -Compare,
# the paired 5-fold comparison runs (promoted against defaults) for the same pairs.
#
#   .\imputation_studies.ps1              # run the six studies (about nineteen hours)
#   .\imputation_studies.ps1 -DryRun      # only print what would run
#   .\imputation_studies.ps1 -Compare     # the twelve comparison runs (about two hours)
#
# Both modes write to the shared mlflow.db: they are the experiment. The studies are
# independent, so a subset can run elsewhere with the same seed:
#   .\imputation_studies.ps1 -Datasets @("spambase")

[CmdletBinding()]
param(
    [string[]]$Datasets = @("credit-g", "kr-vs-kp", "spambase"),
    [string[]]$Variants = @("20nan", "40nan"),
    [int]$Trials = 40,
    [int]$Seed = 42,
    [string]$Scheduler = "cosine",
    [int]$CvFolds = 5,
    [switch]$Compare,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
# A study that fails must not abort the ones after it (PowerShell 7.4 would otherwise
# turn a non-zero exit code into a terminating error under "Stop").
$PSNativeCommandUseErrorActionPreference = $false
Set-Location $PSScriptRoot

function Get-PromotedPath([string]$dataset) {
    $base = $dataset.Split("_")[0]
    # Nested Join-Path: the three-argument form needs PowerShell 7, this runs on 5.1 too.
    return Join-Path (Join-Path "datasets/hiperparams" $base) "$dataset.imputation.json"
}

function Invoke-Trainer([string[]]$trainerArgs) {
    $uvArgs = @("run", "--python", "3.10", "python", "main.py") + $trainerArgs
    Write-Host "uv $($uvArgs -join ' ')"
    if ($DryRun) { return "DRY" }
    & uv @uvArgs
    if ($LASTEXITCODE -eq 0) { return "OK" } else { return "FAILED ($LASTEXITCODE)" }
}

$results = @()
$sweepStart = Get-Date
foreach ($base in $Datasets) {
    foreach ($variant in $Variants) {
        $dataset = "${base}_${variant}"
        $promoted = Get-PromotedPath $dataset

        if (-not $Compare) {
            Write-Host ""
            Write-Host "=== $dataset | reduced study, $Trials trials ===" -ForegroundColor Cyan
            $start = Get-Date
            $status = Invoke-Trainer @(
                "--dataset_name", $dataset, "--task", "imputation", "--use_optuna",
                "--search_space", "reduced", "--n_trials", $Trials,
                "--lr_scheduler", $Scheduler, "--seed", $Seed, "--promote_best"
            )
            $elapsed = (Get-Date) - $start
            $results += [pscustomobject]@{
                Dataset = $dataset; Run = "study"; Status = $status
                Minutes = [math]::Round($elapsed.TotalMinutes, 1)
            }
            Write-Host "[$status] $dataset / study in $([math]::Round($elapsed.TotalMinutes, 1)) min"
            continue
        }

        # The comparison: the same run twice, once reading the promoted file and once
        # with it moved aside so the lookup falls through to the defaults (or to the
        # shared file, if one exists: params.config_source on the run says which).
        $common = @(
            "--dataset_name", $dataset, "--task", "imputation", "--cv_folds", $CvFolds,
            "--lr_scheduler", $Scheduler, "--seed", $Seed
        )
        if (-not $DryRun -and -not (Test-Path $promoted)) {
            Write-Host "[skip] ${dataset}: no promoted configuration at $promoted; run the study first" -ForegroundColor Yellow
            $results += [pscustomobject]@{ Dataset = $dataset; Run = "promoted"; Status = "SKIPPED"; Minutes = 0 }
            continue
        }

        Write-Host ""
        Write-Host "=== $dataset | promoted ($promoted) ===" -ForegroundColor Cyan
        $start = Get-Date
        $status = Invoke-Trainer $common
        $elapsed = (Get-Date) - $start
        $results += [pscustomobject]@{
            Dataset = $dataset; Run = "promoted"; Status = $status
            Minutes = [math]::Round($elapsed.TotalMinutes, 1)
        }
        Write-Host "[$status] $dataset / promoted in $([math]::Round($elapsed.TotalMinutes, 1)) min"

        Write-Host ""
        Write-Host "=== $dataset | defaults (promoted file moved aside) ===" -ForegroundColor Cyan
        $aside = "$promoted.aside"
        Write-Host "[setup] $promoted -> $aside"
        if (-not $DryRun) { Move-Item $promoted $aside -Force }
        try {
            $start = Get-Date
            $status = Invoke-Trainer $common
            $elapsed = (Get-Date) - $start
        }
        finally {
            # Put the promoted file back even on Ctrl+C.
            if (-not $DryRun -and (Test-Path $aside)) { Move-Item $aside $promoted -Force }
            Write-Host "[cleanup] restored $promoted"
        }
        $results += [pscustomobject]@{
            Dataset = $dataset; Run = "defaults"; Status = $status
            Minutes = [math]::Round($elapsed.TotalMinutes, 1)
        }
        Write-Host "[$status] $dataset / defaults in $([math]::Round($elapsed.TotalMinutes, 1)) min"
    }
}

if ($results.Count -gt 0) {
    Write-Host ""
    $mode = if ($Compare) { "COMPARISON" } else { "STUDIES" }
    Write-Host "$mode SUMMARY (total $([math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)) min)" -ForegroundColor Green
    $results | Format-Table -AutoSize
    if ($Compare) {
        Write-Host "Compare in MLflow: tags.run_role = 'parent', tags.task = 'imputation', tags.is_optuna = 'false', tags.lr_scheduler = '$Scheduler', grouped by params.config_source"
        Write-Host "Metric: cv/test/impute/induced/impute_score/mean with ci95_lower / ci95_upper (lower is better; 1.0 is baseline parity)"
    } else {
        Write-Host "Studies in MLflow: tags.run_role = 'optuna_study', tags.task = 'imputation', tags.search_space = 'reduced'"
        Write-Host "Promoted files: datasets/hiperparams/<base>/<base>_<variant>.imputation.json (commit them)"
    }
}

param(
    [string]$At = "02:00",
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$nightly = Join-Path $PSScriptRoot "run-compat-nightly.ps1"
$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$nightly`" -TimeoutSeconds $TimeoutSeconds"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask `
    -TaskName "Raeen Compatibility Nightly" `
    -Description "Measured Astro/Minecraft/UE5/small-title compatibility and audio/input/UI acceptance." `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Force | Out-Null

Write-Host "Installed 'Raeen Compatibility Nightly' for $At."

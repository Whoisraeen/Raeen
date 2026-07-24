param(
    [int]$TimeoutSeconds = 180,
    [string]$Library = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    cargo build --release -p raeen-gui
    if ($Library) {
        cargo xtask compat discover --library $Library
    } else {
        cargo xtask compat discover
    }
    cargo xtask compat run --tier nightly --profile max-fps --timeout $TimeoutSeconds
    cargo xtask acceptance run
    cargo xtask compat publish
} finally {
    Pop-Location
}

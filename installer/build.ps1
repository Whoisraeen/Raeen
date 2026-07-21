<#
    build.ps1 - build the XPS5X release, then package it two ways:

      1. dist\XPS5X-<version>-Setup.exe        (Inno Setup installer)
      2. dist\XPS5X-<version>-portable-win64.zip (portable, extract-and-run)

    The installer step needs Inno Setup 6 (ISCC.exe). If it isn't found the
    script still produces the portable ZIP and tells you how to get Inno.

    Usage (from anywhere):
      powershell -ExecutionPolicy Bypass -File installer\build.ps1
      installer\build.ps1 -SkipBuild            # reuse target\release\xps5x.exe
      installer\build.ps1 -Version 0.2.0        # override the version string
      installer\build.ps1 -SkipInstaller        # portable ZIP only
#>

[CmdletBinding()]
param(
    [switch]$SkipBuild,      # reuse an existing target\release\xps5x.exe
    [switch]$SkipAssets,     # don't regenerate branding art
    [switch]$SkipInstaller,  # skip the Inno step (portable ZIP only)
    [switch]$SkipPortable,   # skip the portable ZIP
    [string]$Version         # override the auto-detected version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ------------------------------------------------------------------ paths ---
$InstallerDir = $PSScriptRoot
$RepoRoot     = Split-Path -Parent $InstallerDir
$AssetsDir    = Join-Path $InstallerDir 'assets'
$DistDir      = Join-Path $RepoRoot 'dist'
$ReleaseExe   = Join-Path $RepoRoot 'target\release\xps5x.exe'

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "    $msg"   -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg"   -ForegroundColor Yellow }

# ---------------------------------------------------------------- version ---
if (-not $Version) {
    $cargo = Get-Content (Join-Path $RepoRoot 'Cargo.toml') -Raw
    $m = [regex]::Match($cargo, '(?ms)\[workspace\.package\].*?version\s*=\s*"([^"]+)"')
    if (-not $m.Success) { throw "Could not read version from Cargo.toml [workspace.package]." }
    $Version = $m.Groups[1].Value
}
Write-Step "XPS5X packaging - version $Version"

if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir | Out-Null }

# ----------------------------------------------------------------- assets ---
if (-not $SkipAssets -or -not (Test-Path (Join-Path $AssetsDir 'xps5x.ico'))) {
    Write-Step "Generating branding assets"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $InstallerDir 'scripts\gen-assets.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Asset generation failed." }
    Write-Ok "assets ready"
}

# ------------------------------------------------------------------ build ---
if (-not $SkipBuild) {
    Write-Step "cargo build --release -p xps5x-gui"
    Push-Location $RepoRoot
    try {
        & cargo build --release -p xps5x-gui
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed." }
    } finally {
        Pop-Location
    }
}
if (-not (Test-Path $ReleaseExe)) {
    throw "Release binary not found at $ReleaseExe. Run without -SkipBuild first."
}
Write-Ok "binary: $ReleaseExe ($([math]::Round((Get-Item $ReleaseExe).Length/1MB,1)) MB)"

# --------------------------------------------------------------- portable ---
# A self-contained folder that runs from anywhere and writes everything beside
# itself - the natural fit for the app's relative-path model. Games go in the
# bundled Games\ folder (config.toml points there).
if (-not $SkipPortable) {
    Write-Step "Building portable ZIP"
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ("xps5x-portable-" + [System.Guid]::NewGuid().ToString('N'))
    $root  = Join-Path $stage 'XPS5X'
    New-Item -ItemType Directory -Path (Join-Path $root 'themes\default') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $root 'Games')          -Force | Out-Null

    Copy-Item $ReleaseExe                                       (Join-Path $root 'xps5x.exe')
    Copy-Item (Join-Path $AssetsDir 'xps5x.ico')               (Join-Path $root 'xps5x.ico')
    Copy-Item (Join-Path $RepoRoot 'themes\default\theme.toml') (Join-Path $root 'themes\default\theme.toml')
    Copy-Item (Join-Path $RepoRoot 'LICENSE')                  (Join-Path $root 'LICENSE.txt')
    Copy-Item (Join-Path $RepoRoot 'THIRD_PARTY_NOTICES.md')   (Join-Path $root 'THIRD_PARTY_NOTICES.md')
    Copy-Item (Join-Path $RepoRoot 'README.md')               (Join-Path $root 'README.md')

    # Portable config: scan the bundled Games\ folder next to the exe.
    $portableConfig = @"
# XPS5X portable configuration. Paths are relative to this folder.
# Drop each game in its own sub-folder under Games\ (containing an eboot.bin).

[general]
fullscreen = true
window_width = 1920
window_height = 1080
vsync = true
selected_theme = "default"

[graphics]
backend = "Vulkan"
resolution_scale = 1.0
shader_cache = true
gpu_device_index = 0
validation_layers = false

[audio]
enabled = true
volume = 1.0
spatial_audio = true

[input]
dualsense_features = true
deadzone = 0.15

[debug]
logging = true
log_level = "info"
dump_gpu_commands = false
dump_shaders = false
trace_syscalls = false

[paths]
games_dir = "games"
firmware_dir = "firmware"
save_dir = "savedata"
shader_cache_dir = "shader_cache"
log_dir = "logs"
game_folders = ["Games"]
key_provider_path = ""
"@
    # UTF-8 without BOM (Rust's config loader reads UTF-8; a BOM would break TOML).
    [System.IO.File]::WriteAllText((Join-Path $root 'config.toml'), $portableConfig, (New-Object System.Text.UTF8Encoding($false)))
    Set-Content -Path (Join-Path $root 'Games\PUT-GAMES-HERE.txt') -Value 'Put each game in its own sub-folder here (each containing an eboot.bin).' -Encoding UTF8
    Set-Content -Path (Join-Path $root 'READ-ME-FIRST.txt') -Value @"
XPS5X (portable) $Version

1. Keep this whole folder together.
2. Put your games under Games\  (one sub-folder per game, each with an eboot.bin).
3. Run xps5x.exe.

Settings, logs, saves and shader cache are all written inside this folder, so
you can move or delete it cleanly. Needs the Microsoft Visual C++ 2015-2022 x64
runtime (most systems already have it; the installer edition installs it for you).
"@ -Encoding UTF8

    $zip = Join-Path $DistDir "XPS5X-$Version-portable-win64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    # Build entries by hand with explicit forward-slash names. Under Windows
    # PowerShell 5.1 (.NET Framework 4.x) BOTH Compress-Archive and
    # ZipFile.CreateFromDirectory write backslash separators, which violate the
    # ZIP spec and confuse extractors like 7-Zip. Naming each entry ourselves is
    # the only reliable way to get spec-compliant forward slashes here.
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $fs = [System.IO.File]::Open($zip, [System.IO.FileMode]::CreateNew)
    $archive = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in (Get-ChildItem -Path $root -Recurse -File)) {
            $rel = $f.FullName.Substring($stage.Length + 1).Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $f.FullName, $rel, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally {
        $archive.Dispose(); $fs.Dispose()
    }
    Remove-Item $stage -Recurse -Force
    Write-Ok "portable: $zip ($([math]::Round((Get-Item $zip).Length/1MB,1)) MB)"
}

# -------------------------------------------------------------- installer ---
function Find-ISCC {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

if (-not $SkipInstaller) {
    Write-Step "Building Inno Setup installer"
    $iscc = Find-ISCC
    if (-not $iscc) {
        Write-Warn2 "Inno Setup 6 (ISCC.exe) not found - skipping the installer."
        Write-Warn2 "Install it, then re-run:  winget install --id JRSoftware.InnoSetup"
        Write-Warn2 "Download:                 https://jrsoftware.org/isdl.php"
    } else {
        Write-Ok "using $iscc"
        & $iscc "/DMyAppVersion=$Version" (Join-Path $InstallerDir 'xps5x.iss')
        if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)." }
        $setup = Join-Path $DistDir "XPS5X-$Version-Setup.exe"
        if (Test-Path $setup) {
            Write-Ok "installer: $setup ($([math]::Round((Get-Item $setup).Length/1MB,1)) MB)"
        }
    }
}

# ---------------------------------------------------------------- summary ---
Write-Step "Done. Artifacts in $DistDir :"
Get-ChildItem $DistDir -File | Where-Object { $_.Name -like "XPS5X-$Version-*" } |
    ForEach-Object { Write-Host ("    " + $_.Name + "  (" + [math]::Round($_.Length/1MB,1) + " MB)") }

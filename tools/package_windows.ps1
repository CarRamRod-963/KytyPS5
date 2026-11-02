<#
.SYNOPSIS
    Packs a Windows release of the KytyPS5 Yotei fork into one zip: the launcher with its Qt runtime, the emulator, the
    emulator's assets, the shader hints, the licences, the readme and the player guide.

.DESCRIPTION
    The readme is README-YOTEI.md under its own name: the documents linking to it and those it links to keep their
    paths (docs\, LICENSES\).
    The emulator and the launcher are built beforehand (the launcher build folder holds launcher.exe after windeployqt).
    The zip holds no game files, no user data (saves, caches, settings, logs) and no debug files; the script stops when
    the staging folder holds any of them.

.EXAMPLE
    pwsh tools/package_windows.ps1 -Emulator build\kyty_emulator.exe -LauncherDir build_launcher\launcher `
        -Hints hints\PPSA26344_01.512.000.kyhints -Guide KytyPS5-Yotei-Player-Guide.pdf -Version v0.1 -Out dist
#>
param(
    [Parameter(Mandatory = $true)][string]$Emulator,    # kyty_emulator.exe of the release build
    [Parameter(Mandatory = $true)][string]$LauncherDir, # the launcher build folder (launcher.exe, Qt DLLs and plugins)
    [string]$Hints = '',                                # a shader hints file (.kyhints), shipped in hints\
    [string]$Guide = '',                                # the player guide (PDF), shipped next to the readme
    [string]$Version = 'v0.1',
    [string]$Out = 'dist'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$name = "KytyPS5-Yotei-$Version-windows-x64"
$stage = Join-Path $Out $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

# The launcher and its Qt runtime (what windeployqt put next to it), without build files.
$launcherFiles = @('launcher.exe', 'libwinpthread-1.dll') + (Get-ChildItem $LauncherDir -Filter 'Qt6*.dll' | ForEach-Object Name)
foreach ($file in $launcherFiles) { Copy-Item (Join-Path $LauncherDir $file) $stage }
foreach ($plugins in 'generic', 'iconengines', 'imageformats', 'networkinformation', 'platforms', 'styles', 'tls') {
    $source = Join-Path $LauncherDir $plugins
    if (Test-Path $source) { Copy-Item -Recurse $source (Join-Path $stage $plugins) }
}

# The emulator and its assets (as the CMake install rules: cursors, fonts, the loading art, the trophy sound).
Copy-Item $Emulator (Join-Path $stage 'kyty_emulator.exe')
$assets = Join-Path $stage 'assets'
New-Item -ItemType Directory -Force (Join-Path $assets 'images'), (Join-Path $assets 'sounds') | Out-Null
Copy-Item -Recurse (Join-Path $repo 'assets\cursors') (Join-Path $assets 'cursors')
Copy-Item -Recurse (Join-Path $repo 'assets\fonts') (Join-Path $assets 'fonts')
Copy-Item (Join-Path $repo 'assets\images\loading-valley.jpg') (Join-Path $assets 'images')
Copy-Item (Join-Path $repo 'assets\sounds\trophy-unlock.wav') (Join-Path $assets 'sounds')

# The shader hints: imported by the launcher on the game's first start (hints\<title id>_<version>.kyhints).
if ($Hints -ne '') {
    New-Item -ItemType Directory -Force (Join-Path $stage 'hints') | Out-Null
    Copy-Item $Hints (Join-Path $stage 'hints')
}

# Licences, the readme and the documents it links to (the same relative paths as in the source tree).
Copy-Item (Join-Path $repo 'LICENSE') $stage
Copy-Item -Recurse (Join-Path $repo 'LICENSES') (Join-Path $stage 'LICENSES')
Copy-Item (Join-Path $repo 'README-YOTEI.md') $stage
if ($Guide -ne '') { Copy-Item $Guide (Join-Path $stage 'KytyPS5-Yotei-Player-Guide.pdf') }
Copy-Item (Join-Path $repo 'CONTRIBUTING.md') $stage
New-Item -ItemType Directory -Force (Join-Path $stage 'docs\launcher') | Out-Null
foreach ($doc in 'PERFORMANCE.md', 'FIXES.md', 'FRAME.md', 'ARCHITECTURE.md', 'TESTING.md', 'CODES.md', 'launcher\DESIGN-SPEC.md') {
    Copy-Item (Join-Path $repo "docs\$doc") (Join-Path $stage "docs\$doc")
}

# Nothing that must not ship: game files, user data, caches, logs, debug files.
$forbidden = Get-ChildItem -Recurse -File $stage | Where-Object {
    $_.Extension -in '.pdb', '.map', '.ilk', '.profraw', '.profdata', '.journal', '.bin', '.elf', '.self', '.sprx', '.prx', '.pkg' -or
    $_.Name -in 'eboot.bin', 'param.sfo', 'param.json', 'Kyty.ini', 'imgui.ini' -or
    $_.FullName -match '\\(_SaveData|_PipelineCache|_TempData|_DownloadData|_Logs|logs|notes|sce_sys)\\'
}
if ($forbidden) {
    $forbidden | ForEach-Object { Write-Host "not for the package: $($_.FullName)" }
    throw 'the package holds files it must not ship'
}

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "PACKAGE_DONE $zip ($size MB, $((Get-ChildItem -Recurse -File $stage).Count) files)"

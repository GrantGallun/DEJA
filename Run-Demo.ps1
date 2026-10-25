param([int]$Port = 8766, [string]$Video = '', [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$demoCandidates = @()
if ($env:SMARTREPLAY_PYTHON) { $demoCandidates += $env:SMARTREPLAY_PYTHON }
$demoCandidates += @(Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
$demoCommand = Get-Command python -ErrorAction SilentlyContinue
if ($demoCommand) { $demoCandidates += $demoCommand.Source }
# Microsoft Store Python keeps packages here; a matching local interpreter can use them.
$demoPackagePaths = @(Get-ChildItem -Path "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.*\LocalCache\local-packages\Python*\site-packages" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
$demoOriginalPythonPath = $env:PYTHONPATH
$demoPython = $null
foreach ($demoCandidate in ($demoCandidates | Select-Object -Unique)) {
    foreach ($demoPackages in (@('') + $demoPackagePaths)) {
        $env:PYTHONPATH = if ($demoPackages) { "$demoPackages;$demoOriginalPythonPath" } else { $demoOriginalPythonPath }
        try {
            & $demoCandidate -c 'import cv2, numpy, imageio_ffmpeg; import smartreplay.demo.server' 2>$null
            if ($LASTEXITCODE -eq 0) { $demoPython = $demoCandidate; break }
        } catch { }
    }
    if ($demoPython) { break }
}
if (!$demoPython) { throw 'Install Python and run: python -m pip install -r requirements-demo.txt. Or set SMARTREPLAY_PYTHON to your Python executable.' }
$demoArgs = @('-m', 'smartreplay.demo', '--port', $Port)
if ($Video) { $demoArgs += @('--video', $Video) }
if ($NoBrowser) { $demoArgs += '--no-browser' }
Write-Host "Starting Smart Replay at http://127.0.0.1:$Port (Ctrl+C to stop)."
try { & $demoPython @demoArgs } finally { $env:PYTHONPATH = $demoOriginalPythonPath }

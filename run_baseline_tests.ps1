param([string]$PythonPath)

$ErrorActionPreference = 'Stop'

if (-not $PythonPath) {
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $existingPython = Join-Path $env:USERPROFILE 'anaconda3\envs\memory_agent\python.exe'
    if (Test-Path -LiteralPath $projectPython) {
        $PythonPath = $projectPython
    } elseif (Test-Path -LiteralPath $existingPython) {
        $PythonPath = $existingPython
    } else {
        throw 'Provide -PythonPath for a project environment with requirements.txt installed.'
    }
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path

$testEnvironment = @{
    PYTHONIOENCODING = 'utf-8'
    HF_HUB_OFFLINE = '1'
    TRANSFORMERS_OFFLINE = '1'
    HF_HUB_DISABLE_TELEMETRY = '1'
}
$previousEnvironment = @{}
$results = @()

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($name in $testEnvironment.Keys) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $testEnvironment[$name], 'Process')
    }
    Write-Host "Python: $PythonPath"
    foreach ($module in @(
        'test_contextual_memory_flow',
        'test_ollama_timing',
        'test_response_pipeline_mock',
        'test_memory_v2_interface_mock',
        'test_agent_memory_versions'
    )) {
        Write-Host "Running: python -m unittest $module -v"
        & $PythonPath -m unittest $module -v
        $results += [PSCustomObject]@{ Module = $module; ExitCode = $LASTEXITCODE }
    }
} finally {
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
    Pop-Location
}

$results | Format-Table -AutoSize
if (@($results | Where-Object { $_.ExitCode -ne 0 }).Count -gt 0) {
    exit 1
}
exit 0

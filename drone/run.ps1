# Launcher for Windows PowerShell. Usage, from anywhere:
#     .\run.ps1 <script> [script options]
#     .\run.ps1 03_collect_data.py --name run1 --hold 10
#     .\run.ps1 01_inspect_env.py --headless
#
# It moves to your IsaacLab folder and runs the script with Isaac Lab's own Python (via `uv`).
# If scripts are blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# ======================= EDIT THIS =======================
# Folder where YOU installed Isaac Lab (the folder that contains pyproject.toml).
# You can also set the ISAACLAB_DIR environment variable instead.
$IsaacLabDir = if ($env:ISAACLAB_DIR) { $env:ISAACLAB_DIR } else { "D:\isaac\IsaacLab" }
# =========================================================

if ($args.Count -lt 1) {
    Write-Host "Usage: .\run.ps1 <script> [options] (e.g. .\run.ps1 03_collect_data.py --name run1)"
    exit 1
}
if (-not (Test-Path (Join-Path $IsaacLabDir "pyproject.toml"))) {
    Write-Host "ERROR: Isaac Lab not found at: $IsaacLabDir"
    Write-Host "Edit `$IsaacLabDir at the top of run.ps1 (or set the ISAACLAB_DIR environment variable)."
    exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'uv' is not installed on your machine. Install it from https://docs.astral.sh/uv/ or run pip install uv."
    exit 1
}

$script = Join-Path $PSScriptRoot $args[0]
$rest = @()
if ($args.Count -gt 1) { $rest = $args[1..($args.Count - 1)] }
Push-Location $IsaacLabDir
try { uv run --extra isaacsim python $script @rest }
finally { Pop-Location }

# Launch hik_camera_ui: conda activate OCRInkjetCoderV2, then python.
# Prerequisite (once): conda init powershell  OR  ensure conda.exe path below exists.

$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\OCRInkjetCoder_V3"

function Initialize-CondaForSession {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        (& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
        return
    }
    $condaExeCandidates = @(
        @(
            "$env:CONDA_EXE"
            "D:\anaconda\Scripts\conda.exe"
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
            "C:\ProgramData\miniconda3\Scripts\conda.exe"
        ) | Where-Object { $_ -and (Test-Path $_) }
    )

    if ($condaExeCandidates.Count -lt 1) {
        Write-Error "conda.exe not found. Install conda, run 'conda init powershell', or edit run_hik_camera_ui.ps1 to add your Scripts\conda.exe path."
    }
    $condaExe = $condaExeCandidates[0]
    (& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
}

Initialize-CondaForSession
conda activate OCRInkjetCoderV2
Set-Location $ProjectRoot
python "$ProjectRoot\hik_camera_ui.py" --auto-connect --hardware-trigger

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$logDir = "outputs\full\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "c2_iid_training.log"

Write-Host "Lancement C2 IID (federated IID)..."
Write-Host "Logs ecrits dans: $logFile"
Write-Host "Ouvre ce fichier pour suivre la progression."
Write-Host ""

& .\.venv\Scripts\python.exe scripts\train_federated.py --config configs\feastnet_full_iid.yaml 2>&1 |
    Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Termine. Resultats dans outputs\full\reports\federated_iid_results.json"

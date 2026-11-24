# Rehearsal helper: wipe the demo case(s) without a new backup, then record again.
# The worker must be stopped first and started again afterwards so it rebuilds the FAISS index.
$root = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $root
& .venv\Scripts\python.exe scripts\demo\reset_cases.py --no-backup
Pop-Location

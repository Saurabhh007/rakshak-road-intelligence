# RAKSHAK - CREATE ONE-COMMAND DEMO LAUNCHER
# Usage:
# .\start_rakshak.ps1 live      (Laptop Webcam - Primary Live Mode)
# .\start_rakshak.ps1 webcam    (Laptop Webcam)
# .\start_rakshak.ps1 fallback  (Prerecorded Video)

$Mode = $args[0]

if ($args.Count -ne 1 -or ($Mode -ne "live" -and $Mode -ne "webcam" -and $Mode -ne "fallback")) {
    Write-Host "Usage:"
    Write-Host ".\start_rakshak.ps1 live      (Laptop Webcam - Live Mode)"
    Write-Host ".\start_rakshak.ps1 webcam    (Laptop Webcam)"
    Write-Host ".\start_rakshak.ps1 fallback  (Prerecorded Video)"
    exit 1
}

# Resolve the project root directory
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Get-Location
}

# Determine video source based on the mode
if ($Mode -eq "live" -or $Mode -eq "webcam") {
    $videoSource = "0"
    
    Write-Host ""
    Write-Host "RAKSHAK LAPTOP WEBCAM LIVE MODE"
    Write-Host "Camera: Laptop Webcam (Index 0)"
    Write-Host "Backend: http://localhost:8000"
    Write-Host "Frontend: http://localhost:5173"
    Write-Host "YOLO: REAL"
    Write-Host "Fallback: AVAILABLE"
    Write-Host ""
}
else {
    $videoSource = "ai/test_videos/road_video.mp4"
    
    Write-Host ""
    Write-Host "RAKSHAK FALLBACK MODE"
    Write-Host "Video: $videoSource"
    Write-Host "Backend: http://localhost:8000"
    Write-Host "Frontend: http://localhost:5173"
    Write-Host "YOLO: REAL"
    Write-Host ""
}

# Set environment variables for the parent process
$env:PYTHONPATH = "backend;."
$env:VIDEO_SOURCE = $videoSource
if ($Mode -eq "live" -or $Mode -eq "webcam") {
    $env:CAMERA_URL = $videoSource
}

# Launch Backend in a new window
# We set the console title and run the backend using uvicorn
$backendCommand = "[Console]::Title = 'RAKSHAK Backend'; `$env:PYTHONPATH='backend;.'; `$env:VIDEO_SOURCE='$videoSource'; `$env:CAMERA_URL='$videoSource'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand -WorkingDirectory $scriptDir

# Launch Frontend in a new window
# We set the console title and run the frontend Vite dev server
$frontendCommand = "[Console]::Title = 'RAKSHAK Frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand -WorkingDirectory "$scriptDir\frontend"

# Automatically open http://localhost:5173 after a short delay
Start-Sleep -Seconds 3
Write-Host "Opening frontend in your browser..."
Start-Process "http://localhost:5173"

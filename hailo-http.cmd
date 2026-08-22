@echo off
REM hailo-http.cmd -- start the Hailo-8L HTTP sidecar the offload-harness calls.
REM Pins the venv interpreter and PYTHONPATH; the harness spawns this on demand
REM and the process exits itself after HAILO_SIDECAR_IDLE_SEC (default 300).
set "HAILO_ROOT=%~dp0"
set "HAILO_ROOT=%HAILO_ROOT:~0,-1%"
set "PYTHONPATH=%HAILO_ROOT%\shared"
if not defined HAILO_MODELS_DIR set "HAILO_MODELS_DIR=D:\Dev\hailo-models"
if not defined HAILO_VISION_ENABLED set "HAILO_VISION_ENABLED=1"
"%HAILO_ROOT%\.venv\Scripts\python.exe" "%HAILO_ROOT%\server\http_server.py" %*

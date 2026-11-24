@echo off
setlocal

rem Starts the API, the GPU worker and the Vite dev server, each in its own console window.
rem Environment variables are read from .env in the project root (git-ignored), for example:
rem   ATMCCTV_SECRET=<random 32+ chars>
rem   ATMCCTV_ADMIN_PASSWORD=<first admin password, only used while the user table is empty>
rem If ATMCCTV_SECRET is missing, a random one is generated and saved to .env.

set "ROOT=%~dp0"

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo Python venv not found at "%ROOT%.venv". Run the setup steps in README.md first.
  pause
  exit /b 1
)

where npm >nul 2>nul || (
  echo npm not found on PATH. Install Node.js first.
  pause
  exit /b 1
)

if exist "%ROOT%.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%.env") do set "%%A=%%B"
)

if not defined ATMCCTV_SECRET (
  echo ATMCCTV_SECRET not set. Generating one and saving it to .env
  for /f %%S in ('powershell -NoProfile -Command "$b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); [BitConverter]::ToString($b).Replace('-','')"') do (
    set "ATMCCTV_SECRET=%%S"
    >>"%ROOT%.env" echo(
    >>"%ROOT%.env" echo ATMCCTV_SECRET=%%S
  )
)

if not defined ATMCCTV_SECRET (
  echo Could not generate ATMCCTV_SECRET. Add it to .env manually.
  pause
  exit /b 1
)

if not exist "%ROOT%frontend\node_modules" (
  echo Installing frontend dependencies
  pushd "%ROOT%frontend"
  call npm install
  popd
)

start "ATMCCTV API" /D "%ROOT%." cmd /k .venv\Scripts\python.exe -m uvicorn atmcctv.api.main:app --host 127.0.0.1 --port 8000

rem Give the API a moment to create/migrate the database before the worker opens it.
timeout /t 3 /nobreak >nul

start "ATMCCTV Worker" /D "%ROOT%." cmd /k .venv\Scripts\python.exe -m atmcctv.worker
start "ATMCCTV Frontend" /D "%ROOT%frontend" cmd /k npm run dev

echo.
echo API:      http://127.0.0.1:8000  (serves frontend\dist)
echo Frontend: http://127.0.0.1:5173  (Vite dev server, proxies /api)
echo Close each window or press Ctrl+C in it to stop that process.

endlocal

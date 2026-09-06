@echo off
REM Exports the game to SampleGame\build\CampusNPC.exe.
REM
REM Requires Godot 4.7 export templates. If the export fails with
REM "No export template found", install them once from the editor
REM (Editor > Manage Export Templates > Download and Install), or drop the
REM contents of Godot_v4.7-stable_export_templates.tpz into
REM   %APPDATA%\Godot\export_templates\4.7.stable\
REM
REM The .exe is the game client only. NPC replies come from
REM backend\gguf_server.py -- see tools\run_demo.bat.

setlocal
set "GODOT=E:\Godot\Godot_v4.7-stable_win64_console.exe"
set "PROJECT=%~dp0.."
set "OUT=%~dp0..\..\build\CampusNPC.exe"

if not exist "%GODOT%" (
    echo [!] Godot not found at %GODOT%
    echo     Edit GODOT at the top of this script to point at your install.
    pause
    exit /b 1
)

if not exist "%~dp0..\..\build" mkdir "%~dp0..\..\build"

echo Exporting to %OUT% ...
"%GODOT%" --headless --path "%PROJECT%" --export-release "Windows Desktop" "%OUT%"
if errorlevel 1 (
    echo [!] Export failed.
    pause
    exit /b 1
)

echo.
echo Built:
dir /b "%~dp0..\..\build"
endlocal

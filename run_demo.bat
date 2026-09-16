@echo off
REM Runs the playable NPC demo: model server (voices, player memory) + game.
REM
REM Double-click this from the project root, or run it from anywhere; every
REM path below is resolved relative to this file.
REM
REM The exported .exe is only the game client. NPC replies come from
REM backend/gguf_server.py, which loads the Q4_K_M adapter models from
REM training/gguf_models/ (about 5 GB, deliberately not packed into the exe),
REM speaks with the Piper voices in backend/voices/, and learns about the
REM player with the fact extractor in training/extractor/player_facts_gliner/.
REM Without the server the campus still loads and is walkable; the dialogue
REM box just reports that it is offline.
REM
REM Closing the game also stops the server this script started. A server that
REM was already running before this script is reused and left running.

setlocal
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
set "GAME_DIR=%REPO_ROOT%\SampleGame\new-game-project"
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
set "GAME_EXE=%REPO_ROOT%\SampleGame\build\CampusNPC.exe"
set "VOICES=%REPO_ROOT%\backend\voices"
set "EXTRACTOR=%REPO_ROOT%\training\extractor\player_facts_gliner"
set "HEALTH=http://127.0.0.1:8000/health"
set "SERVER_TITLE=NPC model server"

if not exist "%PYTHON%" (
    echo [!] No virtualenv at %PYTHON%
    echo     Create it and install requirements.txt from the project root first.
    pause
    exit /b 1
)

REM --- voices ---------------------------------------------------------------
REM Speech is optional at runtime, but without voice models the NPCs are
REM silently mute, which is easy to mistake for a bug. Fetch them once.
if not exist "%VOICES%\*.onnx" (
    echo No NPC voices found. Downloading them once, about 315 MB...
    "%PYTHON%" "%REPO_ROOT%\backend\fetch_voices.py"
    if errorlevel 1 (
        echo [!] Voice download failed. The demo will run without speech.
    )
)

REM --- speech input -----------------------------------------------------------
REM Hold Tab in a conversation to talk. Whisper base.en (~290 MB) transcribes it
REM on the server; fetch it once so talking does not silently do nothing.
"%PYTHON%" -c "from huggingface_hub import snapshot_download as s; s('openai/whisper-base.en', allow_patterns=['*.json','*.txt','model.safetensors'], local_files_only=True)" >nul 2>&1
if errorlevel 1 (
    echo Speech input model not found. Downloading whisper-base.en once, about 290 MB...
    "%PYTHON%" -c "from huggingface_hub import snapshot_download as s; s('openai/whisper-base.en', allow_patterns=['*.json','*.txt','model.safetensors'])"
    if errorlevel 1 echo [!] Download failed. You can still type to NPCs.
)

REM --- player-fact extractor --------------------------------------------------
REM The fine-tuned weights are git-ignored. Without them the server falls back
REM to the zero-shot base model, which misses much of how students type
REM ("i am class cse d"), so say so here instead of letting it look broken.
if not exist "%EXTRACTOR%\intent_head.pt" (
    echo [!] Fine-tuned player-fact extractor not found at
    echo     %EXTRACTOR%
    echo     NPCs will understand less of what you tell them. To build it, about 5 minutes on a GPU:
    echo       .venv\Scripts\python.exe training\extractor\make_extraction_data.py
    echo       .venv\Scripts\python.exe training\extractor\train_extractor.py
    echo       .venv\Scripts\python.exe training\extractor\make_intent_data.py
    echo       .venv\Scripts\python.exe training\extractor\train_intent.py
    echo.
)

REM --- game build -----------------------------------------------------------
REM Rebuild when the exe is missing, or older than any NPC script. This check
REM exists because it bit once already: speech was added to the game code but
REM the exe was not re-exported, so the demo launched a build that never asked
REM for audio even though the server was ready to provide it.
set "STALE=0"
if not exist "%GAME_EXE%" (
    set "STALE=1"
) else (
    powershell -NoProfile -Command ^
      "$exe = (Get-Item '%GAME_EXE%').LastWriteTime;" ^
      "$src = Get-ChildItem '%GAME_DIR%\Systems\NPC' -File | Where-Object { $_.LastWriteTime -gt $exe };" ^
      "if ($src) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "STALE=1"
)
if "%STALE%"=="1" (
    echo Game build is missing or out of date. Rebuilding...
    call "%GAME_DIR%\tools\build.bat"
    if not exist "%GAME_EXE%" (
        echo [!] Build failed; see the output above.
        pause
        exit /b 1
    )
)

REM --- server -----------------------------------------------------------------
REM Reuse a server that is already up: starting a second one would fail to bind
REM port 8000 and leave an error window behind.
set "STARTED_SERVER=0"
curl -s -o nul -m 2 %HEALTH%
if not errorlevel 1 (
    echo A model server is already running on port 8000; using it.
    goto ready
)

echo Starting model server...
pushd "%REPO_ROOT%"
start "%SERVER_TITLE%" "%PYTHON%" -m uvicorn backend.gguf_server:app --port 8000
popd
set "STARTED_SERVER=1"

echo Waiting for the server to come up...
REM Startup builds PDM v2 references for all 8 archetypes, warms every voice
REM (~2 s each on first use) and loads the player-fact extractor, so expect
REM about 15-20 seconds before the first health check answers.
set /a TRIES=0
:wait
set /a TRIES+=1
if %TRIES% gtr 120 (
    echo [!] Server did not respond after 120s. Launching anyway - NPCs will
    echo     report the connection failure in-game.
    goto launch
)
curl -s -o nul -m 2 %HEALTH%
if errorlevel 1 (
    REM One-second pause. Not `timeout /t 1`: timeout refuses to run when
    REM stdin is not an interactive console ("Input redirection is not
    REM supported") and returns immediately, so launched from a shortcut or
    REM another tool the loop would burn all tries in a few seconds and start
    REM the game before the server was up. ping works everywhere.
    ping -n 2 127.0.0.1 >nul
    goto wait
)
echo Server is up.

:ready
REM Say what is actually on, rather than leaving a mute NPC or a forgetful one
REM to be discovered in-game.
powershell -NoProfile -Command ^
  "$h = Invoke-RestMethod %HEALTH%;" ^
  "if ($h.speech -and $h.voices.Count -gt 0) { Write-Host ('Speech:          ON  (' + $h.voices.Count + ' voices)') }" ^
  "else { Write-Host 'Speech:          OFF - NPCs reply in text only. Run backend\fetch_voices.py to enable.' }" ^
  "if ($h.fact_extractor -like '*player_facts_gliner*') { Write-Host 'Player memory:   ON  (fine-tuned extractor)' }" ^
  "elseif ($h.fact_extractor) { Write-Host 'Player memory:   PARTIAL - base extractor only, see the note above.' }" ^
  "else { Write-Host 'Player memory:   OFF - no extractor model available.' }" ^
  "if ($h.speech_to_text) { Write-Host ('Talking:         ON  (hold Tab in a conversation; ' + $h.speech_to_text + ')') }" ^
  "else { Write-Host 'Talking:         OFF - whisper model missing, typing only. See backend\stt.py.' }"

:launch
echo Launching the game. Close it to finish.
start "" /wait "%GAME_EXE%"

if "%STARTED_SERVER%"=="1" (
    echo Game closed. Stopping the model server...
    taskkill /FI "WINDOWTITLE eq %SERVER_TITLE%*" /T /F >nul 2>&1
)
endlocal

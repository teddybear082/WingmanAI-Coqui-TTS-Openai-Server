@echo off
setlocal enabledelayedexpansion

echo Welcome to the Openai-Compatible Server for Coqui-TTS!
echo .
:: Prompt for language
echo Select a language:
echo 1. English (en)
echo 2. Spanish (es)
echo 3. French (fr)
echo 4. German (de)
echo 5. Italian (it)
echo 6. Portuguese (pt)
echo 7. Polish (pl)
echo 8. Turkish (tr)
echo 9. Russian (ru)
echo 10. Dutch (nl)
echo 11. Czech (cs)
echo 12. Arabic (ar)
echo 13. Chinese (zh-cn)
echo 14. Japanese (ja)
echo 15. Hungarian (hu)
echo 16. Korean (ko)
echo 17. Hindi (hi)
set /p lang_choice="Input the number for your language choice and press enter: "

:: Map language choice to code
set "language_id="
if "%lang_choice%"=="1"  set language_id=en
if "%lang_choice%"=="2"  set language_id=es
if "%lang_choice%"=="3"  set language_id=fr
if "%lang_choice%"=="4"  set language_id=de
if "%lang_choice%"=="5"  set language_id=it
if "%lang_choice%"=="6"  set language_id=pt
if "%lang_choice%"=="7"  set language_id=pl
if "%lang_choice%"=="8"  set language_id=tr
if "%lang_choice%"=="9"  set language_id=ru
if "%lang_choice%"=="10" set language_id=nl
if "%lang_choice%"=="11" set language_id=cs
if "%lang_choice%"=="12" set language_id=ar
if "%lang_choice%"=="13" set language_id=zh-cn
if "%lang_choice%"=="14" set language_id=ja
if "%lang_choice%"=="15" set language_id=hu
if "%lang_choice%"=="16" set language_id=ko
if "%lang_choice%"=="17" set language_id=hi

if "%language_id%"=="" (
    echo Invalid language selection.
    goto :eof
)

:: Prompt for hardware
echo.
set /p gpu_choice="Run with GPU or CPU? (type 'gpu' or 'cpu' and press enter): "
set "args_common=--model_path xtts_model\main --config_path xtts_model\main\config.json --speakers_file_path xtts_model\main\speakers.json --language_idx %language_id%"

set "args="
if /i "%gpu_choice%"=="gpu" (
    set "args=--use_cuda --lowvram !args_common!"
) else if /i "%gpu_choice%"=="cpu" (
    set "args=!args_common!"
) else (
    echo Invalid choice. Please type 'gpu' or 'cpu'.
    goto :eof
)
:: Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Run the server in the current terminal
echo.
echo Starting the TTS server...
echo Access a test webpage by CTRL clicking this link after it loads: http://localhost:5002
echo.

:: Run the Python server
python server-with-low-vram-and-stream.py !args!

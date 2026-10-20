@echo off
setlocal
for /f "tokens=2 delims=:." %%C in ('chcp') do set "ORIGINAL_CP=%%C"
chcp 65001 >nul
cd /d "%~dp0"
title AI 토론 경쟁 모델

rem ===================================================================
rem  AI 토론 경쟁 모델 실행기 - 더블클릭하면 아래 순서로 진행합니다.
rem   1) .env가 없으면 만들어서 메모장으로 열기
rem   2) Python 3.10 이상 찾기
rem   3) .venv 가상환경 만들기, requirements.txt가 바뀌었을 때만 패키지 설치
rem   4) 서버 실행 후 준비되면 브라우저 열기
rem  chcp 65001 줄보다 앞에는 한글을 쓰면 안 됩니다. (UTF-8, BOM 없음, CRLF로 저장)
rem ===================================================================

if not exist "app.py" goto :missing_app
if not exist "requirements.txt" goto :missing_app

if exist ".env" goto :env_ready
if exist ".env.example" (
    copy /y ".env.example" ".env" >nul
) else (
    > ".env" echo OPENAI_API_KEY=sk-여기에본인키
    >> ".env" echo ANTHROPIC_API_KEY=sk-ant-여기에본인키
    >> ".env" echo GOOGLE_API_KEY=여기에본인키
)
set "ENV_CREATED=1"
echo .env 파일을 만들었습니다. 지금 열리는 메모장에 API 키 3개를 붙여넣고 저장하세요.
echo 그동안 이 창에서는 실행 준비를 계속합니다.
echo.
start "" notepad ".env"
:env_ready

set "VENV_PY=.venv\Scripts\python.exe"
set "PY_MIN_CHECK=import sys; sys.exit(sys.version_info < (3, 10))"
if not exist "%VENV_PY%" goto :create_venv
"%VENV_PY%" -c "%PY_MIN_CHECK%" >nul 2>&1 && goto :venv_ready
echo 기존 가상환경(.venv)을 쓸 수 없어 새로 만듭니다.
rmdir /s /q ".venv"

:create_venv
rem 3.15는 PyYAML의 Windows용 설치 파일이 아직 없어 패키지 설치가 실패할 수 있으므로 3.14 이하를 먼저 찾습니다.
set "BASE_PY="
for %%V in (3.14 3.13 3.12 3.11 3.10) do if not defined BASE_PY py -%%V -c "%PY_MIN_CHECK%" >nul 2>&1 && set "BASE_PY=py -%%V"
if not defined BASE_PY py -3 -c "%PY_MIN_CHECK%" >nul 2>&1 && set "BASE_PY=py -3"
if not defined BASE_PY python -c "%PY_MIN_CHECK%" >nul 2>&1 && set "BASE_PY=python"
if not defined BASE_PY goto :missing_python
echo 가상환경(.venv)을 만드는 중입니다...
%BASE_PY% --version
%BASE_PY% -m venv .venv
if errorlevel 1 goto :venv_failed
:venv_ready

set "STAMP=.venv\requirements.installed"
fc /b "requirements.txt" "%STAMP%" >nul 2>&1 && goto :deps_ready
echo 필요한 패키지를 설치하는 중입니다. 처음 한 번은 몇 분 걸릴 수 있습니다...
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_failed
copy /y "requirements.txt" "%STAMP%" >nul
:deps_ready

if not defined ENV_CREATED goto :run
echo.
echo 메모장에 API 키 3개를 입력하고 저장했으면, 이 창을 클릭한 뒤 아무 키나 누르세요.
pause >nul

:run
echo.
echo 서버를 시작합니다. 준비되면 브라우저가 자동으로 열립니다.
echo 사용하는 동안에는 이 창을 열어 두세요. 이 창을 닫으면 서버도 꺼집니다.
echo.
set "OPEN_BROWSER=1"
"%VENV_PY%" app.py
if errorlevel 1 goto :server_failed
goto :end

:server_failed
echo.
echo 서버가 오류로 종료되었습니다. 위의 메시지를 확인하세요.
echo  - API 키 문제라면: 메모장으로 .env 파일을 열어 키 3개를 확인하세요.
echo  - 포트 사용 중이라면: 이미 열린 실행 창이 있는지 확인하거나 .env에 PORT=5001 을 추가하세요.
goto :end

:missing_app
echo 실행.bat은 app.py, requirements.txt와 같은 폴더에 있어야 합니다.
echo 현재 폴더: %CD%
goto :end

:missing_python
echo Python 3.10 이상을 찾지 못했습니다.
echo 지금 내려받아지는 Python 3.14 설치 파일을 실행하고, 첫 화면에서 "Add python.exe to PATH"를 체크한 뒤 설치하세요.
echo 설치가 끝나면 실행.bat을 다시 실행하세요.
echo 3.15는 일부 패키지가 아직 지원하지 않아 3.14를 권장합니다.
start "" "https://www.python.org/ftp/python/3.14.8/python-3.14.8-amd64.exe"
goto :end

:venv_failed
echo 가상환경(.venv)을 만들지 못했습니다. 위의 오류 메시지를 확인하세요.
goto :end

:pip_failed
echo.
echo 패키지 설치에 실패했습니다. 인터넷 연결을 확인한 뒤 실행.bat을 다시 실행하세요.
echo 계속 실패하면 아래 Python 버전을 확인하세요. 3.15 이상이면 Python 3.14를 설치하고 .venv 폴더를 지운 뒤 다시 실행하세요.
"%VENV_PY%" --version
goto :end

:end
echo.
pause
if defined ORIGINAL_CP chcp %ORIGINAL_CP% >nul 2>&1
endlocal

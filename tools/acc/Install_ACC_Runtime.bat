@echo off
setlocal
call npm.cmd install --prefix "%~dp0runtime" --save-exact @openai/codex@0.172.0 --no-audit --no-fund
exit /b %errorlevel%

@echo off
setlocal enabledelayedexpansion

rem Set the starting number
set /a count=202

rem Change to the current directory
cd /d %~dp0

rem Loop through all JPG files in the current folder
for %%f in (*.jpg) do (
    rem Rename each file
    ren "%%f" "!count!.jpg"
    set /a count+=1
)

echo Done!
pause
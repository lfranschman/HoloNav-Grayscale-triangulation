@echo off
call ../localProjectPath.bat
set _BATCH_PATH=%~dp0
set _BATCH_PATH=%_BATCH_PATH:~0,-1%

rem we have to use cd ... because when we launch the batch as an administrator, the current path is system...
cd /D %_BATCH_PATH%
if not exist generated (
	mkdir "generated"
)
set _BATCH_NAME=%~n0
set _BATCH_NAME=%_BATCH_NAME:~0,-8%
call %_BATCH_NAME%.bat -nopause > generated/stdout%_BATCH_NAME%.%_USER_SPECIFIC%.txt 2> generated/stderr%_BATCH_NAME%.%_USER_SPECIFIC%.txt
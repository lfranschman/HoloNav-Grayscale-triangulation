@echo off
if not "%_GLOBAL_VARIABLES%" == "" (
	goto _exit
)

set _GLOBAL_VARIABLES=defined
set _GLOBAL_VARIABLES_DEBUG=
set _GLOBAL_VARIABLES_ARM64=
set _GLOBAL_VARIABLES_ARM64_DEBUG=

if exist "%~dp0/userSpecific/globalVariables.%_USER_SPECIFIC%.bat" (
	call "%~dp0/userSpecific/globalVariables.%_USER_SPECIFIC%.bat"
) else (
	call "%~dp0/userSpecific/globalVariables.default.bat"
)

:_exit
pushd %0\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
cd build
msbuild .\synergy.sln /p:Configuration=Release

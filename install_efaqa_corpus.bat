@echo off
setlocal
cd /d "%~dp0"

echo === [1/4] Aliyun mirror (HTTPS) ===
python -m pip install -U "efaqa-corpus-zh>=1.2" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
if %ERRORLEVEL% equ 0 goto ok

echo === [2/4] Tsinghua mirror (HTTPS) ===
python -m pip install -U "efaqa-corpus-zh>=1.2" -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
if %ERRORLEVEL% equ 0 goto ok

echo === [3/4] Aliyun mirror (HTTP, bypass some broken TLS paths) ===
python -m pip install -U "efaqa-corpus-zh>=1.2" -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
if %ERRORLEVEL% equ 0 goto ok

echo === [4/4] PyPI + trusted-host (if mirror had no package) ===
python -m pip install -U "efaqa-corpus-zh>=1.2" -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
if %ERRORLEVEL% equ 0 goto ok

echo.
echo Automatic install failed. Use offline .tar.gz (download in browser or another network):
echo   chatoperastore: https://files.pythonhosted.org/packages/b4/a2/3dee3d5480821a09c819d94503ee57582f14e3bd01337afa3eb4cc05e714/chatoperastore-1.2.0.tar.gz
echo   efaqa-corpus:   https://files.pythonhosted.org/packages/19/9c/afeecf94d235bb23ebac4e945d69ff40cfb3315970105e07d303e3ef6a10/efaqa_corpus_zh-1.2.2.tar.gz
echo Then run in this folder:
echo   python -m pip install chatoperastore-1.2.0.tar.gz efaqa_corpus_zh-1.2.2.tar.gz
echo.
echo To stop pip from always using Douban, edit %%APPDATA%%\pip\pip.ini and remove or change index-url.
pause
exit /b 1

:ok
echo efaqa-corpus-zh installed OK.
exit /b 0

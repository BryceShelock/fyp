import os

# 与 GitHub 官方 efaqa-corpus-zh README 一致：在 import efaqa_corpus_zh 之前设置环境变量。
#
# --- pip 安装不到包？---
# 1) 豆瓣等镜像可能未同步本包 → 换源或运行项目里的 install_efaqa_corpus.bat（会依次试阿里云/清华/HTTP/官方+trusted-host）。
# 2) 访问 pypi.org 报 SSLEOFError：多为代理/校园网/杀毒 HTTPS 扫描打断 TLS，可试国内镜像；仍失败用手机热点或浏览器下载离线包：
#      chatoperastore-1.2.0.tar.gz
#      https://files.pythonhosted.org/packages/b4/a2/3dee3d5480821a09c819d94503ee57582f14e3bd01337afa3eb4cc05e714/chatoperastore-1.2.0.tar.gz
#      efaqa_corpus_zh-1.2.2.tar.gz
#      https://files.pythonhosted.org/packages/19/9c/afeecf94d235bb23ebac4e945d69ff40cfb3315970105e07d303e3ef6a10/efaqa_corpus_zh-1.2.2.tar.gz
#    放到本目录后：python -m pip install chatoperastore-1.2.0.tar.gz efaqa_corpus_zh-1.2.2.tar.gz
# 3) 若 pip 总走豆瓣：检查 %APPDATA%\pip\pip.ini 的 index-url，改为阿里云等或删除该项。
#
# --- 环境变量报错「文件名、目录名或卷标语法不正确」？---
# 说明你在 CMD 里输入了 PowerShell 的 $env:... 语法。请在「当前同一种终端」里用对应写法：
#   CMD:        set EFAQA_DL_LICENSE=你的证书ID
#   PowerShell: $env:EFAQA_DL_LICENSE='你的证书ID'
#
# 方式 A — 在本文件填写证书 ID（勿提交到公开仓库）：
#   _LICENSE_IN_SCRIPT = "LTXxxxx"
# 方式 B — 终端先设置再运行（与上表一致，然后）：
#   python download.py
_LICENSE_IN_SCRIPT = None  # 或 str，例如 "LTXxxxx"

if _LICENSE_IN_SCRIPT:
    os.environ["EFAQA_DL_LICENSE"] = str(_LICENSE_IN_SCRIPT).strip()

_licenseid = os.environ.get("EFAQA_DL_LICENSE", "").strip()
if not _licenseid:
    raise SystemExit(
        "EFAQA_DL_LICENSE 未设置或为空。\n"
        "官方要求：把证书标识设为环境变量后再 import 包才会下载。\n"
        "请设置 _LICENSE_IN_SCRIPT，或在 CMD 用 set、在 PowerShell 用 $env:...（勿混用）。\n"
        "若 pip 装不上 efaqa-corpus-zh：运行 install_efaqa_corpus.bat 或见本文件顶部注释（离线 .tar.gz）。"
    )

print("EFAQA_DL_LICENSE=", _licenseid)

import efaqa_corpus_zh  # noqa: E402

"""
入口脚本。

注意：本仓库历史上存在 `gradio.py` 与第三方包 `gradio` 同名冲突的问题。
如果此文件直接 `import gradio as gr` 会触发循环导入并报：
  AttributeError: partially initialized module 'gradio' has no attribute 'Blocks'

因此这里保持为“薄入口”，只调用真正实现放在 `app_gradio.py` 里。
"""

from app_gradio import main


if __name__ == "__main__":
    main()
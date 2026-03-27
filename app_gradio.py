import os
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

# 避免与本地文件 `gradio.py` 同名冲突，先从 import 路径移除当前目录
_CUR_DIR = os.path.abspath(os.path.dirname(__file__))
_ORIG_PATH = list(sys.path)
sys.path = [
    p
    for p in sys.path
    if os.path.abspath(p or os.getcwd()) != _CUR_DIR
]

import gradio as gr

# 恢复原路径，供本项目模块导入
sys.path = _ORIG_PATH

import torch

from multitask_predict import predict_text, load_model


def predict(text: str, model_dir: str, device: str):
    try:
        result = predict_text(
            text=text,
            model_dir=model_dir,
            device=device,
            max_length=128,
            s1_threshold=0.15,
        )
        # 兼容旧版输出（tuple/list）与新版输出（dict）
        if isinstance(result, dict):
            return result
        if isinstance(result, (list, tuple)):
            return {
                "raw_output": list(result),
                "_warning": "模型返回了非 dict 结构，已按兼容模式展示。",
            }
        return {"raw_output": str(result)}
    except Exception as e:
        return {"error": str(e)}


def show_device(model_dir: str, device: str):
    # 触发一次加载并返回实际设备
    _, _, d = load_model(model_dir, device=device)
    return f"当前推理设备：{d}（cuda可用={torch.cuda.is_available()}）"


def fetch_backend_logs(api_base: str, limit: int = 20):
    try:
        base = (api_base or "").strip().rstrip("/")
        if not base:
            return {"error": "请填写后端地址，例如 http://127.0.0.1:8010"}
        query = urllib.parse.urlencode({"limit": int(limit)})
        url = f"{base}/monitor/latest?{query}"
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.URLError as e:
        return {"error": f"连接后端失败: {e}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    with gr.Blocks() as demo:
        gr.Markdown("## 三任务心理建模（S1多标签 / S2单标签 / S3风险检测）")
        with gr.Tabs():
            with gr.TabItem("本地模型调试"):
                model_dir_in = gr.Textbox(
                    value="multitask_output/best_model",
                    label="model_dir",
                    interactive=True,
                )
                device_in = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="device",
                )
                device_info = gr.Textbox(label="设备状态", interactive=False)
                text_in = gr.Textbox(
                    label="输入文本",
                    value="我压力好大，作业又好多，朋友也不想理我，也有点不想活了",
                    lines=3,
                )
                out = gr.JSON(label="预测结果")

                gr.Button("检查设备").click(fn=show_device, inputs=[model_dir_in, device_in], outputs=[device_info])
                btn = gr.Button("预测")
                btn.click(fn=predict, inputs=[text_in, model_dir_in, device_in], outputs=[out])

            with gr.TabItem("后端链路监控"):
                gr.Markdown("显示 `wechat_mock.html -> backend_api.py -> 模型输出/action` 的最新记录。")
                api_base = gr.Textbox(value="http://127.0.0.1:8010", label="backend_api 地址")
                log_limit = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="拉取条数")
                logs_out = gr.JSON(label="最新后端推理日志")
                gr.Button("刷新日志").click(fetch_backend_logs, inputs=[api_base, log_limit], outputs=[logs_out])

    # 仅本地启动：不使用 share（避免 frpc 下载/代理/杀软问题）
    # 某些环境下 Gradio 会误判 localhost 不可访问，这里关闭该自检。
    try:
        import gradio.networking as gr_networking

        gr_networking.url_ok = lambda *_args, **_kwargs: True  # type: ignore[assignment]
    except Exception:
        pass

    def _pick_port(preferred_port: int = 7861) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", preferred_port)) != 0:
                return preferred_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp:
            temp.bind(("127.0.0.1", 0))
            return int(temp.getsockname()[1])

    demo.launch(share=False, server_name="127.0.0.1", server_port=_pick_port(7861))


if __name__ == "__main__":
    main()

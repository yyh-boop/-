import os
import tempfile
import threading
from typing import Dict

import torch
from flask import Flask, jsonify, request

from predict_image_prob import (
    generate_dire_pil,
    load_cls_model,
    load_dire_model,
    predict_from_pil,
)


class InferenceService:
    """模型推理服务，进程启动时一次性加载模型，后续请求复用。"""

    def __init__(self) -> None:
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cls_model_path = os.environ.get(
            "CLS_MODEL_PATH",
            os.path.join(self.base_dir, "new_model", "new_pretrained_mixed"),
        )
        self.dire_model_path = os.environ.get(
            "DIRE_MODEL_PATH",
            os.path.join(self.base_dir, "guided-diffusion", "256x256_diffusion_uncond.pt"),
        )
        self.guided_dir = os.environ.get(
            "GUIDED_DIR",
            os.path.join(self.base_dir, "guided-diffusion"),
        )
        self._lock = threading.Lock()

        print(f"[服务] 加载分类模型: {self.cls_model_path}")
        self.cls_model = load_cls_model(self.cls_model_path, self.device)

        print(f"[服务] 加载 DIRE 扩散模型: {self.dire_model_path}")
        self.dire_model, self.diffusion = load_dire_model(
            self.dire_model_path, self.device, self.guided_dir
        )
        print(f"[服务] 模型加载完成，运行设备: {self.device}")

    def predict(self, image_path: str) -> Dict:
        with self._lock:
            dire_pil = generate_dire_pil(
                image_path, self.dire_model, self.diffusion, self.device
            )
            pred_cls, real_prob, ai_prob = predict_from_pil(
                self.cls_model, self.device, dire_pil
            )
        return {
            "predicted_class": pred_cls,
            "predicted_label": "AI生成" if pred_cls == 1 else "真实图片",
            "ai_probability": round(float(ai_prob), 6),
            "real_probability": round(float(real_prob), 6),
        }


app = Flask(__name__, static_folder='web', static_url_path='')
service = InferenceService()


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查接口。"""
    return jsonify({"ok": True, "device": service.device})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_image():
    """
    图片 AI 概率检测接口。
    请求: multipart/form-data, 字段名 image
    响应: { predicted_class, predicted_label, ai_probability, real_probability }
    """
    if request.method == "OPTIONS":
        return ("", 204)

    if "image" not in request.files:
        return jsonify({"error": "请使用字段名 image 上传图片文件"}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "图片文件名为空"}), 400

    ext = os.path.splitext(image_file.filename)[1] or ".png"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        result = service.predict(tmp_path)
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)

# 微信小程序部署说明（AI 图片概率检测）

## 1. 后端推理服务启动

在项目根目录执行：

```
pip install -r requirements.txt
python wechat_api_server.py
```

启动后默认监听 `0.0.0.0:5000`，两个接口：

| 用途       | 方式 | 路径                |
| ---------- | ---- | ------------------- |
| 健康检查   | GET  | /api/health         |
| 图片检测   | POST | /api/predict        |

上传字段名固定为 `image`，返回 JSON：

```json
{
  "predicted_class": 1,
  "predicted_label": "AI生成",
  "ai_probability": 0.9312,
  "real_probability": 0.0688
}
```

可通过环境变量覆盖模型路径：

- `CLS_MODEL_PATH`   — 分类模型 (.pth)
- `DIRE_MODEL_PATH`  — DIRE 扩散模型 (.pt)
- `GUIDED_DIR`       — guided-diffusion 目录

---

## 2. 微信开发者工具导入

1. 打开微信开发者工具 → 「导入项目」
2. 选择本项目根目录（`project.config.json` 已含 appid `wxb706b896a56cebf5`）
3. 基础库选择 **3.0.0+**，点击导入即可预览

---

## 3. 小程序内配置 API 地址

首页顶部有「后端 API 地址」输入框：

- 开发者工具模拟器：`http://127.0.0.1:5000`
- 真机调试（同 Wi-Fi）：`http://192.168.1.x:5000`（换成后端电脑局域网 IP）
- 正式上线：需配置 **HTTPS 域名** 并在小程序后台添加合法域名白名单

---

## 4. 目录结构

```
├── wechat_api_server.py    ← Flask 推理 API（后端）
├── predict_image_prob.py   ← 核心推理逻辑
├── app.js / app.json / app.wxss  ← 小程序全局配置
├── pages/
│   └── index/
│       ├── index.js        ← 页面逻辑
│       ├── index.wxml      ← 页面结构
│       ├── index.wxss      ← 深蓝科技风样式
│       └── index.json      ← 页面配置
└── sitemap.json
```

---

## 5. 常见问题

**Q: 开发者工具提示"域名不合法"**
A: 开发阶段在「详情 → 本地设置」勾选「不校验合法域名」即可。

**Q: 推理很慢**
A: 首次加载 DIRE 模型耗时约 30-60 秒，之后服务热缓存，后续请求秒级响应；GPU 环境明显更快。

**Q: 手机真机测试上传失败**
A: 确保手机和后端服务器在同一局域网，且关闭服务器防火墙的 5000 端口限制。

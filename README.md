# **RealSemCom**

---

## **项目结构**

```
project_root/
│── models/                      # 模型相关
│   ├── vit_encoder.py           # ViT 编码（冻结）
│   ├── vit_transformer_tx.py    # 发送端 ViT 计算偏位特征
│   ├── vit_transformer_rx.py    # 接收端 ViT 处理偏位特征
│   ├── reconstruction_network.py # ViT + CNN 重建图像
│   ├── channel_model.py         # 信道模型（模拟噪声和衰落）
│── train/                       # 训练相关
│   ├── train.py                 # 训练主文件
│   ├── utils.py                 # Checkpoint 管理 & FAISS 检索
│   ├── logger.py                # TensorBoard 训练日志
│── data/
│   ├── dataset.py               # 数据集加载 & 知识库匹配
│── checkpoints/                  # 自动存储的 Checkpoints
│── logs/                         # TensorBoard 训练日志
│── best_model.pth                 # 训练后最佳模型
│── requirements.txt               # 依赖库
│── README.md                      # 本项目说明
```

---

## **1. 项目介绍**

本项目实现了一个 **端到端的图像语义通信系统**，使用 **ViT（Vision Transformer）** 进行编码、传输和解码，并借助 **共享知识库** 提高数据传输效率。

### **✨ 主要特点**

✅ **共享知识库**：基于 ViT 预编码的数据库，提高通信效率\
✅ **端到端优化**：所有模块可联合训练，优化传输性能\
✅ **信道鲁棒性**：支持 **AWGN（加性高斯白噪声）+ Rayleigh 衰落信道**\
✅ **模型优化**：

- **ViT 计算全局特征**
- **门控机制（Gated Fusion）** 控制 `I_ref` 影响权重
- **残差门控（Residual Gating）** 适应信道误差
- **ViT + CNN** 结合，提高图像重建质量\
  ✅ **TensorBoard 监控训练**\
  ✅ **Checkpoint 断点续训**

---

## **2. 各模块设计**

### **发送端**

1. `vit_encoder.py`（**ViT 编码原始图像，冻结参数**）
   - `I_enc = ViT(image)`
   - **查询知识库**：找到最近邻 `I_ref`
2. `vit_transformer_tx.py`（**计算偏位特征 **）
   - `F_offset = CrossAttention(I_ref, I_enc)`
   - **门控融合（Gated Fusion）** 控制 `I_ref` 影响权重

---

### **传输信道**

1. `channel_model.py`（**信道模型**）
   - **模拟 AWGN + Rayleigh 信道**
   - 只传输 `F_offset`
   - **增加信道噪声**

---

### **接收端**

1. `vit_transformer_rx.py`（**恢复 **）

   - `F_recovered = CrossAttention(I_ref, F_offset)`
   - **残差门控（Residual Gating）** 适应信道误差

2. `reconstruction_network.py`（**重建图像 **）

   - `I_rec = ViT + CNN(F_recovered)`
   - **ViT 负责全局信息，CNN 负责局部细节**

---

## **\U0001F4D6 3. 训练方法**

### **1️⃣ 安装依赖**

```bash
pip install -r requirements.txt
```

### **2️⃣ 运行训练**

```bash
python train/train.py
```

---

## **4. 如何监控训练过程**

```bash
tensorboard --logdir=logs/
```

然后打开浏览器：

```
http://localhost:6006
```

---

## **5. 断点恢复训练**

如果训练中断，重新运行：

```bash
python train/train.py
```

**会自动加载 **** 并继续训练**。

---

## **6. 未来改进方向**

- **支持目标检测（YOLO 处理部件级传输）**
- **尝试扩展 **** 处理图像**
- **优化信道编码（BPSK, QPSK）**

---

## **7. 结论**

✅ **端到端优化的图像语义通信系统**\
✅ **ViT 进行编码 & CNN 进行重建**\
✅ **共享知识库 + 信道优化，提高传输效率**\
✅ **TensorBoard 监控 & Checkpoint 断点训练**\
✅ **代码结构清晰，便于扩展**

---

💡 **感谢使用本项目！如果你有改进建议，欢迎讨论！🚀🔥**


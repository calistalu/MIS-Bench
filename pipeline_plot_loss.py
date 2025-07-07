import json
import matplotlib
matplotlib.use("Agg")          # 1️⃣ 无显示环境专用后端
import matplotlib.pyplot as plt

# 读取日志
with open("./output/checkpoint-1380/trainer_state.json") as f:
    state = json.load(f)

train_loss, eval_loss, steps = [], [], []
for entry in state["log_history"]:
    if "loss" in entry:        # 训练损失
        train_loss.append(entry["loss"])
        steps.append(entry["step"])
    if "eval_loss" in entry:   # 验证损失
        eval_loss.append(entry["eval_loss"])

# 画图
plt.figure(figsize=(10, 5))
plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
plt.plot(steps[-len(eval_loss):], eval_loss, label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

# 2️⃣ 保存为文件（PNG、300 dpi）
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()                    # 释放内存

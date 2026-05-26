import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data provided by user
raw = {
    "dimension": ["verbal_fluency","hope_and_pe","persuasiveness","emotional_expression","wau","empathy","abc","arrr"],
    "ICC_value": [-0.038105563,0.376249052,-0.020182789,-1.24E-15,0.129353807,0.436328806,0.151644062,0.067095397]
}

finetune = {
    "dimension": ["verbal_fluency","hope_and_pe","persuasiveness","emotional_expression","wau","empathy","abc","arrr"],
    "ICC_value": [0.012312577,0.217275665,0.017994133,-0.135380486,-0.121774026,0.037368081,0.075898078,0.111069755]
}

raft = {
    "dimension": ["verbal_fluency","hope_and_pe","persuasiveness","emotional_expression","wau","empathy","abc","arrr"],
    "ICC_value": [-0.006652041,0.507025662,0.336200366,0.170287646,0.314486111,0.355035264,0.418129835,0.391822328]
}

traditional_mlp = {
    "dimension": ["verbal_fluency","hope_and_pe","persuasiveness","emotional_expression","wau","empathy","abc","arrr"],
    "ICC_value": [0.30, 0.56, 0.30, 0.51, 0.38, 0.29, 0.57, 0.34]
}

# Convert to DataFrame
df_raw = pd.DataFrame(raw)
df_ft = pd.DataFrame(finetune)
df_raft = pd.DataFrame(raft)
df_mlp = pd.DataFrame(traditional_mlp)

# Radar chart setup
categories = df_raw["dimension"].tolist()
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Values for each condition
values_raw = df_raw["ICC_value"].tolist() + [df_raw["ICC_value"].tolist()[0]]
values_ft = df_ft["ICC_value"].tolist() + [df_ft["ICC_value"].tolist()[0]]
values_raft = df_raft["ICC_value"].tolist() + [df_raft["ICC_value"].tolist()[0]]
values_mlp = df_mlp["ICC_value"].tolist() + [df_mlp["ICC_value"].tolist()[0]]

# Plot radar chart
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)

# Set radial limits
ax.set_ylim(-0.2, 0.6)

# Plot each curve
ax.plot(angles, values_raw, linewidth=2, linestyle='solid', label="Raw", color="tab:blue")
ax.fill(angles, values_raw, alpha=0.15, color="tab:blue")

ax.plot(angles, values_ft, linewidth=2, linestyle='solid', label="Fine-tune", color="tab:orange")
ax.fill(angles, values_ft, alpha=0.15, color="tab:orange")

ax.plot(angles, values_raft, linewidth=2, linestyle='solid', label="RAFT Fine-tune", color="tab:green")
ax.fill(angles, values_raft, alpha=0.15, color="tab:green")

ax.plot(angles, values_mlp, linewidth=2, linestyle='solid', label="Traditional MLP", color="tab:red")
ax.fill(angles, values_mlp, alpha=0.15, color="tab:red")

# Legend and aesthetics
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11, frameon=False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

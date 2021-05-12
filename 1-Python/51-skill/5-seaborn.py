import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

my_font = fm.FontProperties(fname="/usr/share/fonts/times/times.ttf", size=7.5)
sns.set_theme(style="darkgrid", font_scale=0.625)

plt.figure(figsize=(6, 6))  # 图片大小会与清晰度，曲线细腻程度有关

data_1 = pd.read_csv("log/data.csv")
# data_2 = pd.read_csv("../../data/CERL/CERL_Ant-v2_100.csv")

sns.lineplot(x="step", y="score", data=data_1)
# sns.lineplot(x="step", y="score", data=data_2)

plt.xlabel("million steps")
plt.ylabel("average return")
plt.title("Walker2d-v2", fontproperties=my_font)
plt.tight_layout()
plt.savefig("fig/data.png", dpi=2400)
plt.show()

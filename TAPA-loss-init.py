import pandas as pd
import matplotlib.pyplot as plt

# 从Excel文件中读取数据，指定第一行为标题行
df = pd.read_excel('/home/liu_shuyuan/tapa-attack/TaPA/generate/TAPA-init-loss.xlsx', header=0)

# 提取steps和loss列作为横轴和纵轴数据，并转换为字符串类型
# steps = df[0].astype(str)
# loss = df[1].astype(str)

steps = range(505)  # 0到69的整数序列
loss = df.iloc[0].values  # 提取第二行作为loss值
loss2 = df.iloc[1].values


# 创建线性图
plt.figure(figsize=(10, 6))
plt.plot(steps[::10], loss[::2], marker='o', linestyle='-', color='b', label='No keyword initialization')
plt.plot(steps[::10], loss2[::2], marker='*', linestyle='-', color='r', label='Keyword initialization')

# 添加标题和标签
# plt.title('Loss vs Steps')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
plt.xlabel('Iterative steps',fontsize=20)
plt.ylabel('Loss value',fontsize=20)
# 显示图例
# plt.legend()
plt.legend(fontsize=20)
# 显示图像
# plt.grid(True)
plt.grid(True, which='major', linestyle='--', linewidth=0.2)  # 只显示纵轴的主刻度网格线

plt.yticks([i * 0.5 for i in range(int(max(max(loss), max(loss2)) / 0.5) + 1)],fontsize=16)
# plt.gca().invert_yaxis()
plt.xticks(fontsize=16)

plt.savefig('TAPA-loss_vs_steps_harmful.png')
plt.show()

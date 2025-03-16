import pandas as pd
import matplotlib.pyplot as plt
import yaml

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["hardware"]["mode"]

# Загружаем данные
df = pd.read_parquet("../data_analyze/logs_output/il_logs.parquet")

# --- Построение гистограммы ---
plt.figure(figsize=(10, 6))
plt.hist(df["vx"], bins=config["visualization"]["bins"][mode], alpha=0.6, label="vx (горизонтальная скорость)")
plt.hist(df["vy"], bins=config["visualization"]["bins"][mode], alpha=0.6, label="vy (вертикальная скорость)")
plt.axvline(0, color="black", linestyle="dashed", linewidth=1, label="Нулевая скорость")

# Настройки графика
plt.xlabel("Скорость (м/с)")
plt.ylabel("Количество агентов")
plt.title("Распределение скоростей vx и vy")
plt.legend()
plt.grid()

# Сохраняем график
plt.savefig("images/speed_distribution.png", dpi=300)
plt.show()

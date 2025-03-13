import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_parquet("logs_output/il_logs.parquet")

# --- Построение гистограммы ---
plt.figure(figsize=(10, 6))
plt.hist(df["vx"], bins=50, alpha=0.6, label="vx (горизонтальная скорость)")
plt.hist(df["vy"], bins=50, alpha=0.6, label="vy (вертикальная скорость)")
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

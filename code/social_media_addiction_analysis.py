# Makine Öğrenmesi Tabanlı Sosyal Medya Bağımlılığı Analizi
# Kaynak: StudentSocialMediaKodları.pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# 1. Veri setini yükleme ve tanıma
df = pd.read_csv("Students Social Media Addiction.csv")
display(df.head())
print(df.shape)
print(df.columns)

# 2. Eksik veri kontrolü
print(df.isnull().sum())

# 3. Temel istatistiksel analiz
display(df.describe())

# 4. Korelasyon analizi
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# 5. Bağımlılık skoru dağılımı
plt.figure(figsize=(7, 4))
sns.histplot(df["Addicted_Score"], kde=True)
plt.title("Bağımlılık Skoru Dağılımı")
plt.xlabel("Bağımlılık Skoru")
plt.ylabel("Öğrenci Sayısı")
plt.show()

# 6. Günlük kullanım süresi dağılımı
plt.figure(figsize=(7, 4))
sns.histplot(df["Avg_Daily_Usage_Hours"], kde=True)
plt.title("Günlük Sosyal Medya Kullanım Süresi Dağılımı")
plt.xlabel("Günlük Kullanım Süresi (Saat)")
plt.ylabel("Öğrenci Sayısı")
plt.show()

# 7. Kullanım süresi ve bağımlılık ilişkisi
plt.figure(figsize=(8, 5))
sns.regplot(x="Avg_Daily_Usage_Hours", y="Addicted_Score", data=df)
plt.title("Kullanım Süresi ve Bağımlılık Skoru")
plt.xlabel("Günlük Kullanım Süresi (Saat)")
plt.ylabel("Bağımlılık Skoru")
plt.show()

# 8. Uyku süresi ve bağımlılık ilişkisi
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Sleep_Hours_Per_Night",
    y="Addicted_Score",
    data=df
)
plt.title("Uyku Süresi ve Bağımlılık Skoru")
plt.xlabel("Uyku Süresi (Saat)")
plt.ylabel("Bağımlılık Skoru")
plt.show()

# 9. Aykırı değer analizi
z_skor = np.abs(stats.zscore(df["Addicted_Score"]))
print("Z-Score Aykırı Değer İndeksleri:")
print(np.where(z_skor > 3))

Q1 = df["Addicted_Score"].quantile(0.25)
Q3 = df["Addicted_Score"].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

aykiri_degerler = df[
    (df["Addicted_Score"] < alt_sinir) |
    (df["Addicted_Score"] > ust_sinir)
]
display(aykiri_degerler)

# 10. Basit doğrusal regresyon
Y = df["Addicted_Score"]
X = df[["Avg_Daily_Usage_Hours"]]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# 11. Regresyon çizgisi
tahmin = model.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(df["Avg_Daily_Usage_Hours"], df["Addicted_Score"])
plt.plot(df["Avg_Daily_Usage_Hours"], tahmin)
plt.xlabel("Günlük Kullanım Süresi")
plt.ylabel("Bağımlılık Skoru")
plt.title("Regresyon Çizgisi")
plt.show()

# 12. Çoklu regresyon
Y2 = df["Addicted_Score"]
X2 = df[
    [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media"
    ]
]
X2 = sm.add_constant(X2)
coklu_model = sm.OLS(Y2, X2).fit()
print(coklu_model.summary())

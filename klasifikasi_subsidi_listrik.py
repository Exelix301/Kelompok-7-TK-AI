import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("Household energy bill data.csv")  # Sesuaikan nama file

print("Shape data:", df.shape)
print(df.head())

# 2. Preprocessing
df = df.drop_duplicates()

# === Membuat Label ===
df['kelayakan'] = np.where(
    (df['ave_monthly_income'] < df['ave_monthly_income'].quantile(0.45)) &
    (df['num_people'] >= 4) &
    (df['amount_paid'] > df['amount_paid'].quantile(0.4)),
    1,  # Layak
    0   # Tidak Layak
)

print("\nDistribusi label:")
print(df['kelayakan'].value_counts())

# 3. Pilih Fitur
features = [
    'num_rooms',
    'housearea',
    'is_ac',
    'is_tv',
    'is_flat',
    'num_children',
    'is_urban',
    'num_people'          
]

X = df[features]
y = df['kelayakan']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# HeatMap Corelation
corr = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr, 
    annot=True,          # tampilkan angka korelasi
    cmap='coolwarm',     # warna merah-biru
    fmt='.2f',           # 2 angka di belakang koma
    linewidths=0.5,
    square=True
)
plt.title("Heatmap Corelation", fontsize=14)
plt.tight_layout()
plt.show()

# 8. Prediksi & Evaluasi
y_pred = model.predict(X_test_scaled)

print("\n========== HASIL EVALUASI ==========")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()
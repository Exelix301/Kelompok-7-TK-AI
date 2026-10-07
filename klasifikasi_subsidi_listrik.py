import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("Household energy bill data.csv")  

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

# === 3.5 Heatmap Correlation ===
# Menggabungkan fitur dan variabel target untuk menghitung matriks korelasi
correlation_matrix = df[features + ['kelayakan']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True,        # Menampilkan angka nilai korelasi pada setiap sel
    fmt='.2f',         # Membatasi 2 angka di belakang koma
    cmap='coolwarm',   # Gradasi warna (merah = positif kuat, biru = negatif kuat)
    linewidths=0.5,    # Memberi garis pemisah antar kotak
    vmin=-1, vmax=1    # Menetapkan batas rentang korelasi (-1 sampai 1)
)
plt.title("Heatmap Korelasi Fitur & Kelayakan Subsidi", fontsize=12)
plt.tight_layout()
plt.show()


# 4. Membagi data training dan testing
x_train_fitur, x_test_fitur, y_train_label, y_test_label = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling
scaler = StandardScaler()
X_train_normalisasi = scaler.fit_transform(x_train_fitur)
X_test_normalisasi = scaler.transform(x_test_fitur)

# 6. Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_normalisasi, y_train_label)

# 7. Feature ImportanceS
feature_names = X.columns
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Fitur': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Fitur', data=feature_importance_df, palette='viridis', legend=False, hue=y)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()


# 8. Prediksi & Evaluasi
y_prediksi = model.predict(X_test_normalisasi)

print("\n========== HASIL EVALUASI ==========")
print("Akurasi:", accuracy_score(y_test_label, y_prediksi))
print("\nClassification Report:")
print(classification_report(y_test_label, y_prediksi))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_label, y_prediksi)
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# 9. INFERENSI (PREDIKSI DATA RUMAH TANGGA BARU)

# Input Data Rumah Tangga Baru
new_data = pd.DataFrame([[2, 36, 0, 1, 0, 3, 1, 5]], columns=features)

# Standardisasi data baru
new_data_scaled = scaler.transform(new_data)

# Melakukan Prediksi dan Menghitung Probabilitas
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

# Menampilkan Hasil Prediksi
label_mapping = {0: "Tidak Layak", 1: "Layak"}
hasil_prediksi = label_mapping[prediction[0]]

print("\n========== HASIL PREDIKSI DATA BARU ==========")
print(f"Status Kelayakan : {hasil_prediksi}")
print(f"Probabilitas     : Tidak Layak = {probability[0][0]*100:.1f}%, Layak = {probability[0][1]*100:.1f}%")
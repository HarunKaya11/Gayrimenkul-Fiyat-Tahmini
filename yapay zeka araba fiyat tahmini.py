import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1) VERİYİ YÜKLE
# ================================
df = pd.read_csv("train.csv")

# ================================
# 2) ÖZELLİK SEÇİMİ VE TÜRKÇELEŞTİRME
# ================================
secilen_sutunlar = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'YearBuilt',
    'SalePrice'
]

yeni_df = df[secilen_sutunlar].copy()
yeni_df.columns = [
    'Kalite_Puani', 'Yasam_Alani', 'Garaj_Kapasitesi',
    'Bodrum_Alani', 'Yapim_Yili', 'Fiyat'
]

# ================================
# 3) VERİ TEMİZLİĞİ
# ================================
yeni_df = yeni_df.fillna(yeni_df.mean())

X = yeni_df.drop('Fiyat', axis=1)
y = yeni_df['Fiyat']

# ================================
# 4) EĞİTİM VE TEST AYIRMA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 5) MODEL EĞİTİMİ
# ================================
print("Model eğitiliyor, lütfen bekleyin...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# 6) MODEL SONUÇLARI
# ================================
tahminler = model.predict(X_test)
basari_puani = r2_score(y_test, tahminler)
hata_payi = mean_absolute_error(y_test, tahminler)

print("-" * 30)
print(f"MODEL BAŞARISI (R2): %{basari_puani * 100:.2f}")
print(f"ORTALAMA HATA PAYI: {hata_payi:,.0f} Dolar")
print("-" * 30)

karsilastirma = pd.DataFrame({
    'Gerçek Fiyat': y_test.values,
    'Tahmin': tahminler
})
print("\nÖrnek Karşılaştırma (İlk 5 Ev):")
print(karsilastirma.head().astype(int))


# ===================================================
# 7) KULLANICIDAN VERİ ALARAK TAHMİN YAPAN SİSTEM
# ===================================================
print("\n🏠 EV FİYATI TAHMİN SİSTEMİNE HOŞ GELDİNİZ 🏠")
print("-" * 60)
print("Kalite Puanı Rehberi:")
print("   1-3 : Bakımsız, tadilat gerekir.")
print("   4-6 : Ortalama apartman dairesi.")
print("   7-8 : Lüks sayılabilecek kalite.")
print("   9-10: Ultra lüks seviyede.")
print("-" * 60)

def veri_iste(soru_metni, alt_sinir, ust_sinir):
    while True:
        try:
            deger = float(input(soru_metni))
            if alt_sinir <= deger <= ust_sinir:
                return deger
            else:
                print(f"⚠️ Lütfen {alt_sinir} ile {ust_sinir} arasında bir değer giriniz.")
        except ValueError:
            print("❌ Lütfen sayı giriniz!")

try:
    mevcut_yil = datetime.now().year

    v_kalite = veri_iste("1. Kalite Puanı (1-10): ", 1, 10)
    v_alan_m2 = veri_iste("2. Yaşam Alanı (30-550m² arası bir değer olarak giriniz): ", 30, 550)   
    v_garaj  = veri_iste("3. Garaj Kapasitesi (0-10): ", 0, 10)
    v_bodrum_m2 = veri_iste("4. Bodrum Alanı (0-550m² arası bir değer olarak giriniz): ", 0, 550)
    v_yil    = veri_iste(f"5. Yapım Yılı (1940-{mevcut_yil}): ", 1940, mevcut_yil)

    # ===================================================
    # ft kare - metre kare dönüşümleri
    # ===================================================
    v_alan = v_alan_m2 * 10.76
    v_bodrum = v_bodrum_m2 * 10.76

    girilen_veri = pd.DataFrame({
        'Kalite_Puani': [v_kalite],
        'Yasam_Alani': [v_alan],
        'Garaj_Kapasitesi': [v_garaj],
        'Bodrum_Alani': [v_bodrum],
        'Yapim_Yili': [v_yil]
    })

    tahmin_sonucu = model.predict(girilen_veri)

    ENFLASYON_CARPANI = 2.2
    guncel_fiyat = tahmin_sonucu[0] * ENFLASYON_CARPANI

    print("\n" + "="*50)
    print(f"📅 2010 Şartlarında Değer: {int(tahmin_sonucu[0]):,} $")
    print(f"🚀 2025 Tahmini Değer:     {int(guncel_fiyat):,} $")
    print("="*50)

except Exception as e:
    print(f"Hata oluştu: {e}")

# ===================================================
# 8) GRAFİKLER
# ===================================================
plt.figure(figsize=(14, 6))

# --- GRAFİK 1: Doğruluk ---
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=tahminler, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat ($)')
plt.ylabel('Tahmin ($)')
plt.title(f'Doğruluk Grafiği (R2: %{basari_puani*100:.1f})')

# --- GRAFİK 2: Özellik Önemleri ---
plt.subplot(1, 2, 2)
onem_dereceleri = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

sns.barplot(
    x=onem_dereceleri,
    y=onem_dereceleri.index,
    hue=onem_dereceleri.index,
    palette='viridis',
    legend=False
)

plt.xlabel('Özellik Etki Gücü')
plt.title('Ev Fiyatını En Çok Etkileyen Değişkenler')

plt.tight_layout()
plt.show()

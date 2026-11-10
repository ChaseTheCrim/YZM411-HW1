import sys
import csv
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Dosya yollari
BASE = r"C:\Users\unalo\Desktop\YZM411HW1"
path_normal = BASE + r"\normal.jpg"
path_yellow = BASE + r"\yellow.jpg"
path_shadow = BASE + r"\shadow.jpg"
OUT = BASE + "\\"

print("Surumler -> Python:", sys.version.split()[0],
      "| OpenCV:", cv2.__version__,
      "| NumPy:", np.__version__,
      "| Matplotlib:", matplotlib.__version__)

# =========================================================
# ADIM 4: Goruntunun Sayisal Yapisi (Hedef 1)
# =========================================================
print("\n--- ADIM 4: GORUNTUNUN SAYISAL YAPISI ---")

img_bgr_norm = cv2.imread(path_normal)
img_bgr_yellow = cv2.imread(path_yellow)
img_bgr_shad = cv2.imread(path_shadow)

if img_bgr_norm is None or img_bgr_yellow is None or img_bgr_shad is None:
    raise FileNotFoundError("Gorseller okunamadi, dosya yollarini kontrol edin.")

if not (img_bgr_norm.shape == img_bgr_yellow.shape == img_bgr_shad.shape):
    raise ValueError("Fotograflarin boyutlari birbiriyle ayni degil!")

# OpenCV BGR okudugu icin RGB'ye ceviriyoruz
img_rgb_norm = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2RGB)
img_rgb_yellow = cv2.cvtColor(img_bgr_yellow, cv2.COLOR_BGR2RGB)
img_rgb_shad = cv2.cvtColor(img_bgr_shad, cv2.COLOR_BGR2RGB)

H, W, C = img_rgb_norm.shape

print(f"Goruntu Boyutu (H x W x C) : {H} x {W} x {C}")
print(f"NumPy Veri Turu (dtype)    : {img_rgb_norm.dtype}")
print(f"Piksel Deger Araligi       : [{img_rgb_norm.min()}, {img_rgb_norm.max()}]")
print(f"Erisim Formati             : img[y, x, kanal]")

mid_bgr = img_bgr_norm[H // 2, W // 2].tolist()
mid_rgb = img_rgb_norm[H // 2, W // 2].tolist()
print(f"Ornek merkez piksel BGR: {mid_bgr} -> RGB: {mid_rgb}")

# HSV donusumleri
img_hsv_norm = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2HSV)
img_hsv_yellow = cv2.cvtColor(img_bgr_yellow, cv2.COLOR_BGR2HSV)
img_hsv_shad = cv2.cvtColor(img_bgr_shad, cv2.COLOR_BGR2HSV)


# Cekimler arasinda el/kamera hafif kaydigi icin gunes sembolunun merkezini
# otomatik bulan yardimci fonksiyon (Claude ile eklendi)
def get_sun_center(hsv_img):
    low_y = np.array([18, 100, 100])
    high_y = np.array([35, 255, 255])
    mask = cv2.inRange(hsv_img, low_y, high_y)

    # Sadece figurun oldugu orta bolgeye bakmak icin ust ve alti sifirliyoruz
    mask[:int(H * 0.30)] = 0
    mask[int(H * 0.55):] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        raise RuntimeError("Sari gunes amblemi tespit edilemedi!")

    # 0 arka plan oldugu icin 1'den baslayarak en buyuk alana sahip bileseni sec
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest_idx]
    return int(round(cy)), int(round(cx))


centers = {
    "Normal": get_sun_center(img_hsv_norm),
    "Gölge": get_sun_center(img_hsv_shad),
    "Sarı": get_sun_center(img_hsv_yellow)
}
print("\nHizalama merkezleri (y, x):", centers)


# Tek pikseldeki kamera gurultusunden etkilenmemek icin 5x5 komsuluk ortalamasi
def get_patch_mean(img, cy, cx, r=2):
    patch = img[cy - r : cy + r + 1, cx - r : cx + r + 1]
    mean_val = patch.reshape(-1, 3).mean(axis=0)
    return mean_val.round().astype(int).tolist()


# Gunes merkezini (0,0) referans alarak secmis oldugumuz 3 farkli bolge
offsets = {
    "P1 (Güneş yüzü - Sarı)": (0, 0),
    "P2 (Cübbe - Beyaz/Gri)": (110, 0),
    "P3 (Fayans - Zemin)": (-400, -380),
}

rgb_images = {
    "Normal": img_rgb_norm,
    "Gölge": img_rgb_shad,
    "Sarı": img_rgb_yellow
}

print("\n--- 3 Fiziksel Bolgenin RGB Degerleri (5x5 komsuluk ortalamasi) ---")
print(f"{'Bölge':<24} | {'Normal (y,x)':<14} | {'Normal RGB':<16} | {'Gölge RGB':<16} | {'Sarı RGB':<16}")
print("-" * 92)

pixel_rows = []
for bolge_adi, (dy, dx) in offsets.items():
    olcumler = {}
    for isik in ["Normal", "Gölge", "Sarı"]:
        cy, cx = centers[isik]
        olcumler[isik] = get_patch_mean(rgb_images[isik], cy + dy, cx + dx)

    norm_y = centers["Normal"][0] + dy
    norm_x = centers["Normal"][1] + dx
    print(f"{bolge_adi:<24} | {str((norm_y, norm_x)):<14} | {str(olcumler['Normal']):<16} | {str(olcumler['Gölge']):<16} | {str(olcumler['Sarı']):<16}")
    pixel_rows.append([bolge_adi, norm_y, norm_x, olcumler["Normal"], olcumler["Gölge"], olcumler["Sarı"]])


# =========================================================
# ADIM 5: Renk Esikleri ve Maske Alan Oranlari (Hedef 2)
# =========================================================
print("\n--- ADIM 5: RENK ESIKLERI VE MASKE ALAN ORANLARI ---")

# Normal isik altinda belirlenen sabit esikler (diger isiklarda degistirilmedi)
# cv2.inRange kullanimi: StackOverflow renk maskeleme orneklerinden
lower_rgb = np.array([120, 110, 0], dtype=np.uint8)
upper_rgb = np.array([255, 230, 90], dtype=np.uint8)

lower_hsv = np.array([16, 80, 70], dtype=np.uint8)
upper_hsv = np.array([36, 255, 255], dtype=np.uint8)

print("Sabit RGB esigi:", lower_rgb.tolist(), "->", upper_rgb.tolist())
print("Sabit HSV esigi:", lower_hsv.tolist(), "->", upper_hsv.tolist())

HALF_Y = 101
HALF_X = 91
total_pixels = H * W  # Adim 6 ve tam kare orani icin toplam piksel sayisi

datasets = [
    ("Normal", img_rgb_norm, img_hsv_norm),
    ("Gölge", img_rgb_shad, img_hsv_shad),
    ("Sarı", img_rgb_yellow, img_hsv_yellow)
]

mask_results = []
full_results = []

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig_full, axes_full = plt.subplots(3, 3, figsize=(12, 14))

for i in range(len(datasets)):
    label, im_rgb, im_hsv = datasets[i]
    cy, cx = centers[label]

    # ROI kesimi
    ry1, ry2 = cy - HALF_Y, cy + HALF_Y
    rx1, rx2 = cx - HALF_X, cx + HALF_X
    roi_rgb = im_rgb[ry1:ry2, rx1:rx2]
    roi_hsv = im_hsv[ry1:ry2, rx1:rx2]
    roi_pixels = roi_rgb.shape[0] * roi_rgb.shape[1]

    # ROI icinde maskeleme ve yuzde hesabi
    mask_rgb = cv2.inRange(roi_rgb, lower_rgb, upper_rgb)
    mask_hsv = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)

    ratio_rgb = (np.count_nonzero(mask_rgb) / roi_pixels) * 100.0
    ratio_hsv = (np.count_nonzero(mask_hsv) / roi_pixels) * 100.0
    mask_results.append((label, ratio_rgb, ratio_hsv))

    # ROI gorsellestirme
    axes[i, 0].imshow(roi_rgb)
    axes[i, 0].set_title(f"{label} Orijinal ROI")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask_rgb, cmap="gray")
    axes[i, 1].set_title(f"RGB Maske (%{ratio_rgb:.1f})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(mask_hsv, cmap="gray")
    axes[i, 2].set_title(f"HSV Maske (%{ratio_hsv:.1f})")
    axes[i, 2].axis("off")

    # Ek olcum: Tam karede maskeleme (arka planda esige takilan pikselleri gormek icin)
    full_rgb = cv2.inRange(im_rgb, lower_rgb, upper_rgb)
    full_hsv = cv2.inRange(im_hsv, lower_hsv, upper_hsv)

    out_rgb = full_rgb.copy()
    out_rgb[ry1:ry2, rx1:rx2] = 0
    out_hsv = full_hsv.copy()
    out_hsv[ry1:ry2, rx1:rx2] = 0

    full_rgb_ratio = (np.count_nonzero(full_rgb) / total_pixels) * 100.0
    full_hsv_ratio = (np.count_nonzero(full_hsv) / total_pixels) * 100.0
    outside_rgb_px = np.count_nonzero(out_rgb)
    outside_hsv_px = np.count_nonzero(out_hsv)

    full_results.append((label, full_rgb_ratio, full_hsv_ratio, outside_rgb_px, outside_hsv_px))

    axes_full[i, 0].imshow(im_rgb)
    axes_full[i, 0].set_title(f"{label} Tam Kare")
    axes_full[i, 0].axis("off")

    axes_full[i, 1].imshow(full_rgb, cmap="gray")
    axes_full[i, 1].set_title("RGB Maske (tam kare)")
    axes_full[i, 1].axis("off")

    axes_full[i, 2].imshow(full_hsv, cmap="gray")
    axes_full[i, 2].set_title("HSV Maske (tam kare)")
    axes_full[i, 2].axis("off")

fig.tight_layout()
fig.savefig(OUT + "adim5_maskeler.png", dpi=200)
plt.close(fig)

fig_full.tight_layout()
fig_full.savefig(OUT + "adim5_tam_kare.png", dpi=150)
plt.close(fig_full)

print("\nROI Maske Oranlari:")
for lbl, r_rgb, r_hsv in mask_results:
    print(f"  {lbl:<8} -> RGB ROI: %{r_rgb:.2f} | HSV ROI: %{r_hsv:.2f}")

print("\nTam Kare ve ROI Disi Piksel Sayilari:")
for lbl, fr, fh, o_r, o_h in full_results:
    print(f"  {lbl:<8} -> Tam Kare RGB: %{fr:.3f} (ROI disi: {o_r} px) | Tam Kare HSV: %{fh:.3f} (ROI disi: {o_h} px)")


# =========================================================
# ADIM 6: Parlaklik ve Gama Donusumleri (Hedef 3)
# =========================================================
print("\n--- ADIM 6: PARLAKLIK VE GAMA DONUSUMLERI ---")


# Tasma (overflow) olmamasi icin once float32'ye cevirip np.clip ile [0, 255] araliginda tutuyoruz
def change_brightness(img, beta=40):
    temp = img.astype(np.float32) + beta
    return np.clip(temp, 0, 255).astype(np.uint8)


def adjust_gamma(img, gamma=1.0):
    norm_img = img.astype(np.float32) / 255.0
    corrected = 255.0 * (norm_img ** gamma)
    return np.clip(corrected, 0, 255).astype(np.uint8)


orig_mean = float(np.mean(img_rgb_norm))

img_b40 = change_brightness(img_rgb_norm, beta=40)
mean_b40 = float(np.mean(img_b40))

# +40 eklendiginde 255 sinirini asip kirpilan (clipped) kanal degerlerinin sayisi
clipped_pixels = int(np.count_nonzero(img_rgb_norm.astype(np.float32) + 40 > 255))
clip_percent = (clipped_pixels / (total_pixels * 3)) * 100.0

img_g05 = adjust_gamma(img_rgb_norm, gamma=0.5)
mean_g05 = float(np.mean(img_g05))

img_g20 = adjust_gamma(img_rgb_norm, gamma=2.0)
mean_g20 = float(np.mean(img_g20))

print(f"Orijinal Goruntu Ortalamasi   : {orig_mean:.2f}")
print(f"Parlaklik (+40) Beklenen      : {orig_mean + 40:.2f}")
print(f"Parlaklik (+40) Olculen       : {mean_b40:.2f}")
print(f"255'te Kirpilan Kanal Sayisi  : {clipped_pixels} (%{clip_percent:.2f})")
print(f"Gama (0.5) Ortalamasi         : {mean_g05:.2f}")
print(f"Gama (2.0) Ortalamasi         : {mean_g20:.2f}")

# Föydeki 64 degeri icin saglama
g05_test = 255 * ((64 / 255) ** 0.5)
g20_test = 255 * ((64 / 255) ** 2.0)
print(f"Saglama (giris=64) -> gamma=0.5: {g05_test:.1f}, gamma=2.0: {g20_test:.1f}")

# Gorseller ve histogramlarin cizimi
step6_imgs = [img_rgb_norm, img_b40, img_g05, img_g20]
step6_titles = ["Orijinal", "Parlaklık +40", "Gama 0.5", "Gama 2.0"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for j in range(4):
    axes[0, j].imshow(step6_imgs[j])
    axes[0, j].set_title(step6_titles[j])
    axes[0, j].axis("off")

    # ravel() ile tek boyuta indirip histogram cizdirme (StackOverflow)
    axes[1, j].hist(step6_imgs[j].ravel(), bins=64, range=(0, 255))
    axes[1, j].set_title(f"Histogram - {step6_titles[j]}")

plt.tight_layout()
plt.savefig(OUT + "adim6_donusumler.png", dpi=150)
plt.close()


# =========================================================
# ADIM 7: Kucultme (Downsampling) ve Bilgi Kaybi (Hedef 4)
# =========================================================
print("\n--- ADIM 7: KUCULTME VE BILGI KAYBI ---")

cy, cx = centers["Normal"]
roi = img_rgb_norm[cy - HALF_Y : cy + HALF_Y, cx - HALF_X : cx + HALF_X]
rh, rw, _ = roi.shape

# Boyutu 1/4 oraninda kucultme
new_w = max(1, rw // 4)
new_h = max(1, rh // 4)
target_size = (new_w, new_h)

down_nearest = cv2.resize(roi, target_size, interpolation=cv2.INTER_NEAREST)
down_area = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

# Piksellesmeyi ve kaybi orijinal boyutla karsilastirmak icin tekrar buyutme
up_nearest = cv2.resize(down_nearest, (rw, rh), interpolation=cv2.INTER_NEAREST)
up_area = cv2.resize(down_area, (rw, rh), interpolation=cv2.INTER_NEAREST)


def calc_mae(img1, img2):
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return float(np.mean(diff))


# Goruntudeki kenar/detay enerjisini olcmek icin Laplacian varyansi (SO / PyImageSearch yontemi)
def calc_laplacian_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


mae_nearest = calc_mae(roi, up_nearest)
mae_area = calc_mae(roi, up_area)

lap_orig = calc_laplacian_var(roi)
lap_nearest = calc_laplacian_var(up_nearest)
lap_area = calc_laplacian_var(up_area)

print(f"ROI Boyutu: {rw}x{rh} -> Kucultulmus: {new_w}x{new_h}")
print(f"Orijinal      -> Laplacian Varyansi: {lap_orig:.2f}")
print(f"INTER_NEAREST -> MAE: {mae_nearest:.2f} | Laplacian Varyansi: {lap_nearest:.2f}")
print(f"INTER_AREA    -> MAE: {mae_area:.2f} | Laplacian Varyansi: {lap_area:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(roi)
axes[0].set_title("Orijinal ROI (Detay Bölgesi)")
axes[0].axis("off")

axes[1].imshow(up_nearest)
axes[1].set_title("cv2.INTER_NEAREST (1/4 -> 1/1)")
axes[1].axis("off")

axes[2].imshow(up_area)
axes[2].set_title("cv2.INTER_AREA (1/4 -> 1/1)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT + "adim7_ornekleme.png", dpi=200)
plt.close()


# =========================================================
# Tum Sonuclarin CSV Dosyasina Kaydedilmesi (Claude ile eklendi)
# =========================================================
csv_path = OUT + "sonuc_tablosu.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["ADIM 4", "bolge", "y_normal", "x_normal", "RGB_normal", "RGB_golge", "RGB_Sarı"])
    for row in pixel_rows:
        writer.writerow(["", row[0], row[1], row[2], row[3], row[4], row[5]])
    writer.writerow(["", "hizalama_merkez_y_x", centers["Normal"], centers["Gölge"], centers["Sarı"]])
    writer.writerow([])

    writer.writerow(["ADIM 5", "isik", "rgb_maske_roi_yuzde", "hsv_maske_roi_yuzde"])
    for lbl, r_rgb, r_hsv in mask_results:
        writer.writerow(["", lbl, f"{r_rgb:.2f}", f"{r_hsv:.2f}"])

    writer.writerow(["ADIM 5 (tam kare)", "isik", "rgb_yuzde", "hsv_yuzde", "roi_disi_rgb_px", "roi_disi_hsv_px"])
    for lbl, fr, fh, o_r, o_h in full_results:
        writer.writerow(["", lbl, f"{fr:.3f}", f"{fh:.3f}", o_r, o_h])
    writer.writerow([])

    writer.writerow(["ADIM 6", "olcum", "deger"])
    writer.writerow(["", "orijinal_ortalama", f"{orig_mean:.2f}"])
    writer.writerow(["", "parlaklik40_beklenen", f"{orig_mean + 40:.2f}"])
    writer.writerow(["", "parlaklik40_olculen", f"{mean_b40:.2f}"])
    writer.writerow(["", "kirpilan_kanal_degeri", f"{clipped_pixels:.2f}"])
    writer.writerow(["", "kirpilan_yuzde", f"{clip_percent:.2f}"])
    writer.writerow(["", "gama05_ortalama", f"{mean_g05:.2f}"])
    writer.writerow(["", "gama20_ortalama", f"{mean_g20:.2f}"])
    writer.writerow([])

    writer.writerow(["ADIM 7", "yontem", "MAE", "laplacian_varyans"])
    writer.writerow(["", "orijinal", "", f"{lap_orig:.2f}"])
    writer.writerow(["", "INTER_NEAREST", f"{mae_nearest:.2f}", f"{lap_nearest:.2f}"])
    writer.writerow(["", "INTER_AREA", f"{mae_area:.2f}", f"{lap_area:.2f}"])

print("\nTum gorseller ve 'sonuc_tablosu.csv' basariyla kaydedildi.")
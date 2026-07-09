# Aplikasi Klasifikasi SIBI (Sistem Isyarat Bahasa Indonesia)

Project ini adalah aplikasi Flask yang menggunakan model YOLOv8 (`model/my_model.pt`) untuk mendeteksi/mengklasifikasikan Bahasa Isyarat Indonesia (SIBI) melalui unggah gambar (`/predict-image`) atau pemrosesan frame video secara real-time (`/predict-frame`).

---

## Panduan Hosting ke Render (Native Python)

Untuk menghosting aplikasi ini ke **Render** sebagai **Web Service** menggunakan runtime Python bawaan, ikuti langkah-langkah berikut:

### 1. Persiapan Repository
1. Buat repository baru di GitHub atau GitLab (pastikan private jika tidak ingin model Anda diakses publik).
2. Push seluruh file project ini ke repository Anda (pastikan folder `model/` dan file `model/my_model.pt` ikut ter-upload).

### 2. Buat Web Service Baru di Render
1. Masuk ke dashboard [Render](https://dashboard.render.com).
2. Klik tombol **New +** di pojok kanan atas, lalu pilih **Web Service**.
3. Hubungkan akun GitHub/GitLab Anda dan pilih repository project ini.

### 3. Konfigurasi Web Service
Isi pengaturan Web Service sebagai berikut:

* **Name**: `aplikasi-sibi` (atau nama lain pilihan Anda)
* **Region**: Pilih lokasi terdekat (misalnya Singapore `singapore` untuk performa terbaik di Indonesia)
* **Branch**: `main` (atau cabang git yang Anda gunakan)
* **Runtime**: `Python 3`
* **Build Command**:
  ```bash
  pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
  ```
  > [!IMPORTANT]
  > **SANGAT PENTING**: Jangan gunakan `pip install -r requirements.txt` standar. Perintah di atas memaksa Render untuk menginstal **PyTorch versi CPU saja**. Jika tidak, Render akan mengunduh PyTorch GPU (CUDA) yang ukurannya lebih dari 2GB, yang akan menyebabkan build time-out atau kegagalan ruang disk di tier Free Render.
  
* **Start Command**:
  ```bash
  gunicorn app:app
  ```
  > [!NOTE]
  > Perintah ini menjalankan Flask app secara production-ready menggunakan `gunicorn`. Render secara otomatis mendefinisikan port melalui environment variable `$PORT` dan Gunicorn akan mendeteksi dan mengikat (bind) ke port tersebut secara otomatis.

* **Instance Type**: Pilih **Free** (atau tier berbayar jika ingin resource yang lebih cepat).

### 4. Deploy!
1. Klik tombol **Deploy Web Service** di bagian bawah halaman.
2. Proses build pertama kali mungkin memakan waktu sekitar 3–7 menit karena mengunduh package `ultralytics` dan `torch` CPU. Anda dapat memantau log build di dashboard Render.
3. Setelah status berubah menjadi **Live**, aplikasi Anda dapat diakses melalui URL `.onrender.com` yang disediakan oleh Render.

---

## Jalankan Secara Lokal (Local Development)

Jika ingin menjalankan aplikasi ini di komputer lokal Anda:

1. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   # Aktifkan virtual environment
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Flask:
   ```bash
   python app.py
   ```
4. Buka browser dan akses `http://localhost:5050`.

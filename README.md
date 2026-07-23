# Multi Agent AI Exam Generator

Sistem belajar & latihan soal otomatis berbasis **multi-agent AI Exam Generator**. Cukup masukkan topik yang ingin dipelajari — tanpa perlu mencari sumber sendiri — sistem akan mencari materi di internet, merangkumnya, membuatkan soal latihan, menilai jawaban, dan memberi evaluasi belajar secara otomatis, dalam satu alur yang berulang sampai target nilai tercapai.

https://multi-agent-ai-exam-generators.onrender.com/

Dibangun dengan **Gradio**, **LangGraph**, dan **Groq (Llama 3.3 70B)**, dan siap dideploy ke **Render**.

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Alur Penggunaan](#alur-penggunaan)
- [Arsitektur Multi-Agent](#arsitektur-multi-agent)
- [Sistem Penilaian](#sistem-penilaian)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Instalasi & Menjalankan Secara Lokal](#instalasi--menjalankan-secara-lokal)
- [Environment Variable](#environment-variable)
- [Deploy ke Render](#deploy-ke-render)
- [Struktur Project](#struktur-project)
- [Catatan & Batasan](#catatan--batasan)

---

## Fitur Utama

- **Pencarian & scraping materi otomatis** — pengguna hanya mengetik topik (tanpa URL), agent akan mencari sumber relevan di internet dan mengambil kontennya secara otomatis.
- **Rangkuman materi adaptif level** — materi dirangkum ulang oleh AI sesuai level yang dipilih: Beginner, Medium, atau Advanced, masing-masing dengan gaya bahasa dan kedalaman berbeda.
- **Soal pilihan ganda & essay otomatis** — dibuat berdasarkan materi yang sudah dirangkum, dengan jumlah soal yang bisa diatur (maksimal 25 soal pilihan ganda dan 10 soal essay per sesi).
- **Penilaian otomatis** — pilihan ganda dinilai langsung berdasarkan kunci jawaban, sedangkan essay dinilai oleh AI (dengan feedback singkat) secara paralel agar cepat.
- **Loop latihan otomatis** — jika nilai belum mencapai target, sistem otomatis membuat set soal baru (berbeda dari sebelumnya) untuk percobaan berikutnya, hingga maksimal 5 kali percobaan atau sampai target nilai tercapai.
- **Evaluasi & rekomendasi belajar** — setiap selesai mengerjakan, AI memberi ringkasan performa, kekuatan, kelemahan, dan rekomendasi belajar lanjutan.
- **Antarmuka 3 halaman yang terpisah** — alur belajar dipecah menjadi tiga tahap yang jelas agar tidak membingungkan pengguna.

---

## Alur Penggunaan

Aplikasi dibagi menjadi tiga halaman (tab) yang berurutan:

### 1. Topik & Materi
Pengguna memilih level kesulitan, mengetik topik, dan menentukan jumlah soal pilihan ganda serta essay yang diinginkan. Setelah menekan tombol cari, sistem akan:
1. Mencari sumber di internet berdasarkan topik.
2. Melakukan scraping konten dari beberapa sumber teratas.
3. Merangkum konten tersebut menjadi materi belajar yang rapi.
4. Menyiapkan set soal pertama secara paralel di latar belakang.

Materi rangkuman langsung ditampilkan di halaman ini. Pengguna lalu menekan **Lanjut ke Soal** untuk berpindah ke halaman berikutnya.

### 2. Soal & Jawaban
Menampilkan seluruh soal pilihan ganda dan essay yang telah dibuat berdasarkan materi. Pengguna mengerjakan seluruh soal, lalu menekan tombol untuk mengumpulkan jawaban.

### 3. Nilai & Evaluasi
Menampilkan nilai akhir, rincian jawaban benar/salah untuk pilihan ganda, skor & feedback tiap soal essay, serta evaluasi naratif dari AI. Ada dua kemungkinan hasil:
- **Nilai sudah mencapai target** → sesi latihan selesai.
- **Nilai belum mencapai target** (dan percobaan belum habis) → sistem otomatis menyiapkan set soal baru yang berbeda, dan pengguna bisa menekan **Kerjakan Soal Berikutnya** untuk kembali ke halaman soal.

---

## Arsitektur Multi-Agent

Sistem ini dibangun dengan **LangGraph**, dipecah menjadi tiga graph independen yang masing-masing menjalankan agent-agent-nya secara paralel:

```
┌─────────────────────────┐
│      MATERI GRAPH        │
│                          │
│  search_scraper          │
│        │                │
│        ▼                │
│    summarizer            │
└─────────────────────────┘

┌─────────────────────────┐
│       SOAL GRAPH          │   (berjalan paralel)
│                          │
│  generate_pg   generate_essay
│      (bersamaan, dari START yang sama)
└─────────────────────────┘

┌─────────────────────────┐
│     GRADING GRAPH          │
│                          │
│  grading_pg   grading_essay   (paralel)
│        │           │        │
│        └─────┬─────┘        │
│              ▼               │
│        combine_nilai         │
│              │               │
│              ▼               │
│          evaluation           │
└─────────────────────────┘
```

**Ringkasan tiap agent:**

| Agent | Tugas |
|---|---|
| `search_scraper` | Mencari topik di internet lalu mengambil (scraping) konten dari beberapa sumber teratas |
| `summarizer` | Merangkum konten mentah menjadi materi belajar terstruktur, disesuaikan dengan level |
| `generate_pg` | Membuat soal pilihan ganda baru berdasarkan materi, menghindari soal yang sudah pernah muncul |
| `generate_essay` | Membuat soal essay baru beserta kunci jawaban acuan |
| `grading_pg` | Menilai jawaban pilihan ganda dengan mencocokkan ke kunci jawaban |
| `grading_essay` | Menilai jawaban essay satu per satu secara paralel (multi-thread) menggunakan LLM sebagai penguji |
| `combine_nilai` | Menggabungkan skor pilihan ganda dan essay menjadi nilai akhir berbobot |
| `evaluation` | Menyusun evaluasi naratif: ringkasan performa, kekuatan, kelemahan, dan rekomendasi belajar |

Pemisahan menjadi tiga graph ini membuat proses generate soal (PG + essay) dan proses grading (PG + essay) berjalan **bersamaan**, bukan bergantian, sehingga waktu tunggu pengguna jauh lebih singkat dibanding menjalankannya satu per satu.

---

## Sistem Penilaian

- **Pilihan ganda**: dinilai otomatis, `skor = (jumlah benar / total soal) × 100`.
- **Essay**: setiap jawaban dinilai oleh LLM dengan skor 0–100 beserta feedback singkat, lalu dirata-rata.
- **Nilai akhir**: gabungan berbobot — **70% pilihan ganda** dan **30% essay** (jika salah satu jenis soal tidak diisi, bobot dialihkan sepenuhnya ke jenis yang tersedia).
- **Target kelulusan default**: 80 (dari skala 100).
- **Maksimal percobaan**: 5 kali per topik. Jika nilai belum mencapai target, soal baru yang berbeda otomatis dibuatkan untuk percobaan berikutnya.

Nilai default ini (`TARGET_SKOR_DEFAULT`, `MAX_PERCOBAAN_DEFAULT`) diatur di bagian atas `app.py` dan bisa diubah sesuai kebutuhan.

---

## Teknologi yang Digunakan

| Komponen | Library |
|---|---|
| Antarmuka web | [Gradio](https://gradio.app) |
| Orkestrasi multi-agent | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | [Groq API](https://groq.com) — model `llama-3.3-70b-versatile` (via `langchain-groq`) |
| Pencarian web | `ddgs` (DuckDuckGo Search) |
| Scraping konten | `requests` + `BeautifulSoup` (`lxml` parser) |
| Manajemen env var lokal | `python-dotenv` (opsional) |

---

## Instalasi & Menjalankan Secara Lokal

```bash
# 1. Clone / masuk ke folder project
cd nama-folder-project

# 2. (Opsional tapi disarankan) buat virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Siapkan environment variable
echo "GROQ_API_KEY=isi_api_key_groq_anda" > .env

# 5. Jalankan aplikasi
python app.py
```

Aplikasi akan berjalan di `http://localhost:7860` (atau port yang ditentukan lewat env var `PORT`).

Dapatkan API key gratis di [console.groq.com](https://console.groq.com).

---

## Environment Variable

| Variable | Wajib | Keterangan |
|---|---|---|
| `GROQ_API_KEY` | Ya | API key untuk mengakses model LLM Groq. Aplikasi tidak akan berjalan tanpa ini. |
| `PORT` | Tidak | Port server, otomatis di-set oleh Render. Default lokal: `7860`. |

---

## Deploy ke Render

1. Push project (berisi `app.py`, `requirements.txt`, dan `render.yaml`) ke repository Git.
2. Buat **Web Service** baru di Render, hubungkan ke repository tersebut.
3. Tambahkan environment variable `GROQ_API_KEY` di dashboard Render (Settings → Environment).
4. Render akan otomatis menjalankan `python app.py` sesuai konfigurasi di `render.yaml`.

**Catatan teknis penting:** aplikasi menjalankan `demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)`. Dua hal ini wajib:
- `server_name="0.0.0.0"` — agar bisa diakses dari luar container Render (bukan hanya `127.0.0.1`).
- `ssr_mode=False` — mencegah Gradio membuka proses Node.js terpisah untuk server-side rendering yang bisa bind ke port berbeda dari yang diminta, penyebab umum error *"No open ports detected"* di Render.

---

## Struktur Project

```
.
├── app.py             # Seluruh logika aplikasi: agent, graph, UI Gradio
├── requirements.txt    # Daftar dependency Python
├── render.yaml         # Konfigurasi deployment Render
└── README.md           # Dokumen ini
```

Seluruh logika — agent, graph LangGraph, dan antarmuka Gradio — sengaja disatukan dalam satu `app.py` agar mudah dideploy sebagai satu Web Service tanpa konfigurasi tambahan.

---

## Catatan & Batasan

- Kualitas materi & soal bergantung pada hasil pencarian web saat itu; jika topik terlalu spesifik/niche atau sumber tidak tersedia, scraping bisa gagal dan sistem akan menampilkan pesan error yang jelas.
- Maksimal 3 sumber pertama yang berhasil di-scrape per topik akan dipakai sebagai bahan rangkuman, untuk menjaga kecepatan proses.
- Penilaian essay memakai LLM sebagai penguji — bersifat estimatif, bukan penilaian manusia, sehingga sebaiknya dipakai untuk latihan mandiri, bukan penilaian resmi/akademik.
- Riwayat soal disimpan hanya selama sesi pengguna berlangsung (di memori state Gradio), sehingga akan hilang saat halaman di-refresh atau sesi berakhir.

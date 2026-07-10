---
title: Multi Agent AI Belajar Paralel
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Multi-Agent AI Paralel: Cari Topik, Rangkuman, Soal, Penilaian & Evaluasi

Arsitektur paralel (Groq + LangGraph fan-out/fan-in):
1. Search & Scraper Agent (otomatis dari topik, tanpa URL manual)
2. Summarizer Agent (level-aware: Beginner / Medium / Advanced)
3. Generator Soal PG ‖ Generator Soal Essay (paralel, anti-duplikat antar percobaan)
4. Grading PG ‖ Grading Essay (paralel, essay dinilai paralel per-soal)
5. Combine Nilai (fan-in)
6. Evaluation Agent

## Fitur
- Pilihan **Level**: Beginner, Medium, Advanced
- Input **Topik** bebas — tidak perlu URL, agent mencari & scraping sendiri
- **Loop otomatis**: jika nilai < 80, soal baru otomatis dibuat sampai nilai ≥ 80 atau maksimal 5 percobaan

## Setup
Tambahkan secret `GROQ_API_KEY` di Settings Space ini (dapatkan di https://console.groq.com/keys).

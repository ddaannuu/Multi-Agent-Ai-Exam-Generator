import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Optional

GROQ_MODEL = "llama-3.3-70b-versatile"
TARGET_SKOR_DEFAULT = 80
MAX_PERCOBAAN_DEFAULT = 5

LEVEL_LABEL = {"beginner": "Beginner (Pemula)", "medium": "Medium (Menengah)", "advanced": "Advanced (Lanjutan)"}

LEVEL_INSTRUKSI_MATERI = {
    "beginner": (
        "Gunakan bahasa yang SANGAT SEDERHANA dan mudah dipahami pemula total. "
        "Hindari istilah teknis rumit (jelaskan jika terpaksa dipakai), gunakan banyak contoh konkret sehari-hari, "
        "dan jelaskan dari dasar seolah pembaca baru pertama kali mengenal topik ini."
    ),
    "medium": (
        "Gunakan bahasa baku standar tingkat mahasiswa. Boleh memakai istilah teknis umum dengan penjelasan singkat, "
        "seimbang antara teori dan contoh penerapan."
    ),
    "advanced": (
        "Gunakan bahasa teknis dan mendalam. Asumsikan pembaca sudah familiar dengan konsep dasar bidang ini. "
        "Fokus pada detail, nuansa, perbandingan antar konsep, kelebihan/kekurangan, dan penerapan lanjutan."
    ),
}

LEVEL_INSTRUKSI_SOAL = {
    "beginner": "Soal level DASAR: fokus pada definisi, konsep inti, dan pemahaman umum. Hindari perhitungan atau kasus yang rumit.",
    "medium": "Soal level MENENGAH: kombinasi pemahaman konsep dan penerapan sederhana / studi kasus ringan.",
    "advanced": "Soal level LANJUTAN: menuntut analisis, perbandingan konsep, studi kasus kompleks, dan penerapan mendalam.",
}


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise gr.Error("GROQ_API_KEY belum diset di Space secrets!")
    return ChatGroq(model=GROQ_MODEL, temperature=0.4, api_key=api_key)


def call_llm(system_prompt, user_prompt):
    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return resp.content


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


class AgentState(TypedDict, total=False):
    level: str
    topik: str
    jumlah_pg: int
    jumlah_essay: int
    target_skor: float
    percobaan_ke: int
    max_percobaan: int
    raw_content: str
    materi: str
    sumber: List[str]
    soal_pg: List[Dict]
    soal_essay: List[Dict]
    riwayat_soal_pg: List[str]
    riwayat_soal_essay: List[str]
    jawaban_user_pg: List[str]
    jawaban_user_essay: List[str]
    hasil_pg: Dict
    hasil_essay: Dict
    nilai_akhir: float
    evaluasi: str
    error: str


# ---------------- AGENTS ----------------

def agent_search_scraper(state: AgentState) -> AgentState:
    topik = state.get("topik", "").strip()
    if not topik:
        state["error"] = "Topik tidak diberikan."
        state["raw_content"] = ""
        state["sumber"] = []
        return state

    query = f"{topik} penjelasan lengkap"

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        state["error"] = "Library pencarian (ddgs) belum terpasang."
        state["raw_content"] = ""
        state["sumber"] = []
        return state

    kandidat_url = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="id-id", max_results=8):
                link = r.get("href") or r.get("link") or r.get("url")
                if link:
                    kandidat_url.append(link)
    except Exception:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=8):
                    link = r.get("href") or r.get("link") or r.get("url")
                    if link:
                        kandidat_url.append(link)
        except Exception as e2:
            state["error"] = f"Gagal melakukan pencarian otomatis: {e2}"
            state["raw_content"] = ""
            state["sumber"] = []
            return state

    if not kandidat_url:
        state["error"] = "Tidak ditemukan sumber relevan untuk topik ini."
        state["raw_content"] = ""
        state["sumber"] = []
        return state

    headers = {"User-Agent": "Mozilla/5.0 (compatible; EduScraperBot/1.0)"}
    kumpulan_teks, sumber_berhasil = [], []

    for link in kandidat_url:
        if len(kumpulan_teks) >= 3:
            break
        try:
            resp = requests.get(link, headers=headers, timeout=12)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
                tag.decompose()
            candidates = soup.find_all(["article", "main"])
            text_source = candidates[0] if candidates else soup
            paragraphs = [p.get_text(" ", strip=True) for p in text_source.find_all(["p", "h1", "h2", "h3", "li"])]
            paragraphs = [p for p in paragraphs if len(p) > 30]
            teks = "\n".join(paragraphs) or text_source.get_text(" ", strip=True)
            if len(teks) > 200:
                kumpulan_teks.append(teks[:4000])
                judul = soup.title.get_text(strip=True) if soup.title else link
                sumber_berhasil.append(judul)
        except Exception:
            continue

    raw_text = "\n\n---\n\n".join(kumpulan_teks)
    if not raw_text:
        state["error"] = "Tidak ada sumber yang berhasil di-scraping untuk topik ini."
        state["raw_content"] = ""
        state["sumber"] = []
        return state

    state["raw_content"] = raw_text[:15000]
    state["sumber"] = sumber_berhasil
    state["error"] = ""
    return state


def agent_summarizer(state: AgentState) -> AgentState:
    raw_content = state.get("raw_content", "")
    topik = state.get("topik", "")
    level = state.get("level", "medium")
    if not raw_content:
        state["materi"] = ""
        return state
    instruksi_level = LEVEL_INSTRUKSI_MATERI.get(level, LEVEL_INSTRUKSI_MATERI["medium"])
    system_prompt = (
        "Kamu adalah asisten pendidikan ahli yang merangkum materi belajar dalam Bahasa Indonesia. "
        f"{instruksi_level} Buat rangkuman terstruktur dengan heading & poin-poin. "
        "Jangan tambahkan informasi yang tidak ada di sumber."
    )
    user_prompt = (
        f'Topik yang diminta user: "{topik}"\n\n'
        f'Konten mentah hasil pencarian & scraping otomatis:\n"""\n{raw_content}\n"""\n\n'
        f"Buatkan RANGKUMAN MATERI BELAJAR rapi dalam format Markdown, sesuai level: {LEVEL_LABEL.get(level, level)}."
    )
    state["materi"] = call_llm(system_prompt, user_prompt)
    return state


def agent_generate_pg(state: AgentState) -> Dict:
    materi = state.get("materi", "")
    jumlah_pg = state.get("jumlah_pg", 5)
    level = state.get("level", "medium")
    riwayat = state.get("riwayat_soal_pg", [])
    if not materi:
        return {"soal_pg": []}
    instruksi_level = LEVEL_INSTRUKSI_SOAL.get(level, LEVEL_INSTRUKSI_SOAL["medium"])
    larangan_ulang = ""
    if riwayat:
        daftar = "\n".join(f"- {s}" for s in riwayat[-20:])
        larangan_ulang = f"\n\nPENTING: JANGAN membuat soal yang sama/mirip dengan soal berikut yang sudah pernah diberikan:\n{daftar}\n"
    system_prompt = f"Kamu pembuat soal pilihan ganda yang teliti. {instruksi_level} HANYA balas JSON valid."
    user_prompt = f"""
Berdasarkan MATERI berikut:
\"\"\"
{materi}
\"\"\"
{larangan_ulang}
Buatkan {jumlah_pg} soal PILIHAN GANDA BARU (4 opsi A-D, 1 jawaban benar) sesuai level yang diminta.
Balas HANYA JSON: {{"soal_pg": [{{"soal": "...", "pilihan": {{"A":"...","B":"...","C":"...","D":"..."}}, "jawaban_benar": "A"}}]}}
"""
    raw = call_llm(system_prompt, user_prompt)
    try:
        parsed = extract_json(raw)
        return {"soal_pg": parsed.get("soal_pg", [])}
    except Exception:
        return {"soal_pg": []}


def agent_generate_essay(state: AgentState) -> Dict:
    materi = state.get("materi", "")
    jumlah_essay = state.get("jumlah_essay", 3)
    level = state.get("level", "medium")
    riwayat = state.get("riwayat_soal_essay", [])
    if not materi:
        return {"soal_essay": []}
    instruksi_level = LEVEL_INSTRUKSI_SOAL.get(level, LEVEL_INSTRUKSI_SOAL["medium"])
    larangan_ulang = ""
    if riwayat:
        daftar = "\n".join(f"- {s}" for s in riwayat[-20:])
        larangan_ulang = f"\n\nPENTING: JANGAN membuat soal yang sama/mirip dengan soal berikut yang sudah pernah diberikan:\n{daftar}\n"
    system_prompt = f"Kamu pembuat soal essay yang teliti. {instruksi_level} HANYA balas JSON valid."
    user_prompt = f"""
Berdasarkan MATERI berikut:
\"\"\"
{materi}
\"\"\"
{larangan_ulang}
Buatkan {jumlah_essay} soal ESSAY BARU beserta kunci jawaban acuan.
Balas HANYA JSON: {{"soal_essay": [{{"soal": "...", "kunci_jawaban": "..."}}]}}
"""
    raw = call_llm(system_prompt, user_prompt)
    try:
        parsed = extract_json(raw)
        return {"soal_essay": parsed.get("soal_essay", [])}
    except Exception:
        return {"soal_essay": []}


def agent_grading_pg(state: AgentState) -> Dict:
    soal_pg = state.get("soal_pg", [])
    jawaban_user_pg = state.get("jawaban_user_pg", [])
    detail_pg, benar = [], 0
    for i, soal in enumerate(soal_pg):
        ju = jawaban_user_pg[i] if i < len(jawaban_user_pg) else None
        kunci = soal.get("jawaban_benar")
        is_benar = (ju or "").strip().upper() == (kunci or "").strip().upper()
        if is_benar:
            benar += 1
        detail_pg.append({"soal": soal.get("soal"), "jawaban_user": ju, "jawaban_benar": kunci, "benar": is_benar})
    skor_pg = (benar / len(soal_pg) * 100) if soal_pg else None
    return {"hasil_pg": {"detail": detail_pg, "jumlah_benar": benar, "total": len(soal_pg), "skor": skor_pg}}


def _grade_satu_essay(idx, soal, jawaban_user):
    system_prompt = "Kamu dosen penguji adil. HANYA balas JSON valid."
    user_prompt = f"""
Soal: {soal.get('soal')}
Kunci jawaban: {soal.get('kunci_jawaban')}
Jawaban mahasiswa: \"\"\"{jawaban_user}\"\"\"

Nilai 0-100 berdasarkan kesesuaian & kelengkapan. Balas HANYA JSON:
{{"skor": <0-100>, "feedback": "<1-2 kalimat>"}}
"""
    raw = call_llm(system_prompt, user_prompt)
    try:
        parsed = extract_json(raw)
        skor = float(parsed.get("skor", 0))
        feedback = parsed.get("feedback", "")
    except Exception:
        skor, feedback = 0.0, "Gagal menilai otomatis."
    return idx, {"soal": soal.get("soal"), "jawaban_user": jawaban_user, "skor": skor, "feedback": feedback}


def agent_grading_essay(state: AgentState) -> Dict:
    soal_essay = state.get("soal_essay", [])
    jawaban_user_essay = state.get("jawaban_user_essay", [])
    if not soal_essay:
        return {"hasil_essay": {"detail": [], "skor_rata_rata": None}}
    hasil_sementara = [None] * len(soal_essay)
    with ThreadPoolExecutor(max_workers=min(8, len(soal_essay))) as executor:
        futures = []
        for i, soal in enumerate(soal_essay):
            ju = jawaban_user_essay[i] if i < len(jawaban_user_essay) else ""
            futures.append(executor.submit(_grade_satu_essay, i, soal, ju))
        for future in as_completed(futures):
            idx, hasil = future.result()
            hasil_sementara[idx] = hasil
    skor_list = [h["skor"] for h in hasil_sementara]
    skor_essay_avg = (sum(skor_list) / len(skor_list)) if skor_list else None
    return {"hasil_essay": {"detail": hasil_sementara, "skor_rata_rata": skor_essay_avg}}


def agent_combine_nilai(state: AgentState) -> AgentState:
    hasil_pg = state.get("hasil_pg", {})
    hasil_essay = state.get("hasil_essay", {})
    skor_pg = hasil_pg.get("skor")
    skor_essay = hasil_essay.get("skor_rata_rata")
    komponen = []
    if skor_pg is not None:
        komponen.append((skor_pg, 0.7))
    if skor_essay is not None:
        komponen.append((skor_essay, 0.3))
    if komponen:
        total_bobot = sum(b for _, b in komponen)
        nilai_akhir = sum(s * b for s, b in komponen) / total_bobot
    else:
        nilai_akhir = 0.0
    state["nilai_akhir"] = round(nilai_akhir, 2)
    return state


def agent_evaluation(state: AgentState) -> AgentState:
    target = state.get("target_skor", TARGET_SKOR_DEFAULT)
    percobaan_ke = state.get("percobaan_ke", 1)
    system_prompt = "Kamu mentor belajar suportif dan membangun. Jawab dalam Bahasa Indonesia."
    user_prompt = f"""
Materi:
\"\"\"
{state.get('materi', '')[:3000]}
\"\"\"

Percobaan ke-{percobaan_ke}.
Hasil PG: {json.dumps(state.get('hasil_pg', {}), ensure_ascii=False)}
Hasil Essay: {json.dumps(state.get('hasil_essay', {}), ensure_ascii=False)}
Nilai akhir: {state.get('nilai_akhir')} (target kelulusan: {target})

Tulis EVALUASI (Markdown): 1) Ringkasan performa 2) Kekuatan 3) Kelemahan/miskonsepsi
4) Rekomendasi belajar lanjutan 5) Status: sudah/belum mencapai target {target}, dan bagian mana yang perlu diperkuat jika belum.
"""
    state["evaluasi"] = call_llm(system_prompt, user_prompt)
    return state


# ---------------- GRAPHS (PARALEL) ----------------
materi_builder = StateGraph(AgentState)
materi_builder.add_node("search_scraper", agent_search_scraper)
materi_builder.add_node("summarizer", agent_summarizer)
materi_builder.add_edge(START, "search_scraper")
materi_builder.add_edge("search_scraper", "summarizer")
materi_builder.add_edge("summarizer", END)
materi_graph = materi_builder.compile()

soal_builder = StateGraph(AgentState)
soal_builder.add_node("generate_pg", agent_generate_pg)
soal_builder.add_node("generate_essay", agent_generate_essay)
soal_builder.add_edge(START, "generate_pg")
soal_builder.add_edge(START, "generate_essay")
soal_builder.add_edge("generate_pg", END)
soal_builder.add_edge("generate_essay", END)
soal_graph = soal_builder.compile()

grading_builder = StateGraph(AgentState)
grading_builder.add_node("grading_pg", agent_grading_pg)
grading_builder.add_node("grading_essay", agent_grading_essay)
grading_builder.add_node("combine_nilai", agent_combine_nilai)
grading_builder.add_node("evaluation", agent_evaluation)
grading_builder.add_edge(START, "grading_pg")
grading_builder.add_edge(START, "grading_essay")
grading_builder.add_edge("grading_pg", "combine_nilai")
grading_builder.add_edge("grading_essay", "combine_nilai")
grading_builder.add_edge("combine_nilai", "evaluation")
grading_builder.add_edge("evaluation", END)
grading_graph = grading_builder.compile()


# ---------------- GRADIO APP ----------------
MAX_PG_SLOT = 6
MAX_ESSAY_SLOT = 4


def cari_materi(level_label, topik, jumlah_pg, jumlah_essay):
    if not topik or not topik.strip():
        raise gr.Error("Masukkan topik yang ingin dipelajari terlebih dahulu.")

    level = {v: k for k, v in LEVEL_LABEL.items()}.get(level_label, "medium")

    state: AgentState = {
        "level": level,
        "topik": topik.strip(),
        "jumlah_pg": int(jumlah_pg),
        "jumlah_essay": int(jumlah_essay),
        "target_skor": TARGET_SKOR_DEFAULT,
        "max_percobaan": MAX_PERCOBAAN_DEFAULT,
        "riwayat_soal_pg": [],
        "riwayat_soal_essay": [],
        "percobaan_ke": 0,
    }

    result = materi_graph.invoke(state)
    if result.get("error") and not result.get("materi"):
        raise gr.Error(result["error"])

    materi_md = result.get("materi", "_Materi gagal dibuat._")
    if result.get("sumber"):
        materi_md += "\n\n---\n**Sumber rujukan (dicari & di-scrape otomatis):**\n" + "\n".join(f"- {s}" for s in result["sumber"])

    # generate percobaan pertama langsung
    soal_state = soal_graph.invoke(result)
    result["soal_pg"] = soal_state.get("soal_pg", [])
    result["soal_essay"] = soal_state.get("soal_essay", [])
    result["riwayat_soal_pg"] = [s["soal"] for s in result["soal_pg"]]
    result["riwayat_soal_essay"] = [s["soal"] for s in result["soal_essay"]]
    result["percobaan_ke"] = 1

    pg_updates = []
    soal_pg = result.get("soal_pg", [])
    for i in range(MAX_PG_SLOT):
        if i < len(soal_pg):
            s = soal_pg[i]
            choices = [f'{k}. {v}' for k, v in s["pilihan"].items()]
            pg_updates.append(gr.update(visible=True, label=f"Soal {i+1}: {s['soal']}", choices=choices, value=None))
        else:
            pg_updates.append(gr.update(visible=False))

    essay_updates = []
    soal_essay = result.get("soal_essay", [])
    for i in range(MAX_ESSAY_SLOT):
        if i < len(soal_essay):
            essay_updates.append(gr.update(visible=True, label=f"Essay {i+1}: {soal_essay[i]['soal']}", value=""))
        else:
            essay_updates.append(gr.update(visible=False))

    status_md = f"**Percobaan ke-1** dari maksimal {MAX_PERCOBAAN_DEFAULT} • Target nilai: {TARGET_SKOR_DEFAULT}"

    return (materi_md, result, status_md, "", "", *pg_updates, *essay_updates)


def submit_jawaban(state, *answers):
    if not state:
        raise gr.Error("Silakan cari & rangkum materi terlebih dahulu.")

    n_pg = len(state.get("soal_pg", []))
    n_essay = len(state.get("soal_essay", []))
    pg_answers_raw = answers[:MAX_PG_SLOT]
    essay_answers_raw = answers[MAX_PG_SLOT:MAX_PG_SLOT + MAX_ESSAY_SLOT]

    jawaban_pg = []
    for i in range(n_pg):
        val = pg_answers_raw[i]
        jawaban_pg.append(val.split(".")[0].strip() if val else "")
    jawaban_essay = [essay_answers_raw[i] or "" for i in range(n_essay)]

    state["jawaban_user_pg"] = jawaban_pg
    state["jawaban_user_essay"] = jawaban_essay

    final_state = grading_graph.invoke(state)
    state.update(final_state)

    target = state.get("target_skor", TARGET_SKOR_DEFAULT)
    max_percobaan = state.get("max_percobaan", MAX_PERCOBAAN_DEFAULT)
    percobaan_ke = state.get("percobaan_ke", 1)
    nilai_akhir = final_state.get("nilai_akhir", 0)

    nilai_text = f"## ⭐ Nilai Percobaan ke-{percobaan_ke}: {nilai_akhir} / 100 (target: {target})\n\n"
    hasil_pg = final_state.get("hasil_pg", {})
    if hasil_pg.get("total"):
        nilai_text += f"**Pilihan Ganda:** {hasil_pg['jumlah_benar']} / {hasil_pg['total']} benar (skor {hasil_pg['skor']:.1f})\n\n"
    hasil_essay = final_state.get("hasil_essay", {})
    if hasil_essay.get("skor_rata_rata") is not None:
        nilai_text += f"**Essay (rata-rata):** {hasil_essay['skor_rata_rata']:.1f}\n\n"
        for i, d in enumerate(hasil_essay.get("detail", []), 1):
            nilai_text += f"- Essay {i}: skor {d['skor']:.0f} — _{d['feedback']}_\n"

    evaluasi_md = final_state.get("evaluasi", "")

    lulus = nilai_akhir >= target
    pg_updates = [gr.update() for _ in range(MAX_PG_SLOT)]
    essay_updates = [gr.update() for _ in range(MAX_ESSAY_SLOT)]

    if lulus:
        nilai_text += f"\n\n🎉 **SELAMAT! Nilai sudah mencapai target ({target}). Latihan selesai.**"
        status_md = f"**Status: LULUS ✅** • Percobaan ke-{percobaan_ke} • Nilai {nilai_akhir}"
        for i in range(MAX_PG_SLOT):
            pg_updates[i] = gr.update(interactive=False)
        for i in range(MAX_ESSAY_SLOT):
            essay_updates[i] = gr.update(interactive=False)
    elif percobaan_ke >= max_percobaan:
        nilai_text += f"\n\n⚠️ **Sudah mencapai batas maksimal {max_percobaan} percobaan tanpa mencapai target.** Pelajari kembali evaluasi di atas."
        status_md = f"**Status: BELUM MENCAPAI TARGET ⚠️** • Percobaan ke-{percobaan_ke}/{max_percobaan} (batas maksimal tercapai)"
        for i in range(MAX_PG_SLOT):
            pg_updates[i] = gr.update(interactive=False)
        for i in range(MAX_ESSAY_SLOT):
            essay_updates[i] = gr.update(interactive=False)
    else:
        # ---- Nilai belum mencapai target: buat soal BARU otomatis untuk percobaan berikutnya ----
        nilai_text += f"\n\n📌 **Nilai belum mencapai target {target}. Soal baru sudah disiapkan di bawah — silakan kerjakan lagi.**"
        percobaan_baru = percobaan_ke + 1
        state["percobaan_ke"] = percobaan_baru

        soal_state = soal_graph.invoke(state)
        state["soal_pg"] = soal_state.get("soal_pg", [])
        state["soal_essay"] = soal_state.get("soal_essay", [])
        state["riwayat_soal_pg"] = state.get("riwayat_soal_pg", []) + [s["soal"] for s in state["soal_pg"]]
        state["riwayat_soal_essay"] = state.get("riwayat_soal_essay", []) + [s["soal"] for s in state["soal_essay"]]

        soal_pg = state["soal_pg"]
        for i in range(MAX_PG_SLOT):
            if i < len(soal_pg):
                s = soal_pg[i]
                choices = [f'{k}. {v}' for k, v in s["pilihan"].items()]
                pg_updates[i] = gr.update(visible=True, interactive=True, label=f"Soal {i+1}: {s['soal']}", choices=choices, value=None)
            else:
                pg_updates[i] = gr.update(visible=False)

        soal_essay = state["soal_essay"]
        for i in range(MAX_ESSAY_SLOT):
            if i < len(soal_essay):
                essay_updates[i] = gr.update(visible=True, interactive=True, label=f"Essay {i+1}: {soal_essay[i]['soal']}", value="")
            else:
                essay_updates[i] = gr.update(visible=False)

        status_md = f"**Percobaan ke-{percobaan_baru}** dari maksimal {max_percobaan} • Target nilai: {target}"

    return (state, status_md, nilai_text, evaluasi_md, *pg_updates, *essay_updates)


with gr.Blocks(title="Multi-Agent AI Paralel: Belajar & Latihan Soal") as demo:
    gr.Markdown("# 🎓 Multi-Agent AI Paralel — Cari Topik, Rangkuman, Soal, Penilaian & Evaluasi")
    gr.Markdown(
        "Pilih **level**, ketik **topik** (tanpa perlu URL) — agent akan **mencari & scraping otomatis**. "
        "Arsitektur paralel (Groq + LangGraph fan-out/fan-in): "
        "Search&Scraper → Summarizer → **{Generator PG ‖ Generator Essay}** → **{Grading PG ‖ Grading Essay}** → Combine Nilai → Evaluation. "
        f"Jika nilai belum mencapai **{TARGET_SKOR_DEFAULT}**, soal baru otomatis dibuat sampai maksimal **{MAX_PERCOBAAN_DEFAULT} percobaan**."
    )

    state_box = gr.State(None)

    with gr.Row():
        level_in = gr.Radio(choices=list(LEVEL_LABEL.values()), value=LEVEL_LABEL["medium"], label="Level Kesulitan")
        topik_in = gr.Textbox(label="Topik yang Ingin Dipelajari", placeholder="mis. jenis-jenis machine learning")
    with gr.Row():
        jumlah_pg_in = gr.Slider(1, MAX_PG_SLOT, value=5, step=1, label="Jumlah Soal PG per Sesi")
        jumlah_essay_in = gr.Slider(1, MAX_ESSAY_SLOT, value=3, step=1, label="Jumlah Soal Essay per Sesi")

    btn_cari = gr.Button("🔎 Cari Materi & Buat Soal (Otomatis, Paralel)", variant="primary")

    materi_out = gr.Markdown(label="Materi")
    status_out = gr.Markdown(label="Status Percobaan")

    gr.Markdown("## Soal Pilihan Ganda")
    pg_radios = [gr.Radio(visible=False, label=f"Soal PG {i+1}") for i in range(MAX_PG_SLOT)]

    gr.Markdown("## Soal Essay")
    essay_boxes = [gr.Textbox(visible=False, label=f"Essay {i+1}", lines=3) for i in range(MAX_ESSAY_SLOT)]

    btn_submit = gr.Button(f"✅ Kumpulkan Jawaban & Nilai (Loop otomatis sampai nilai ≥ {TARGET_SKOR_DEFAULT})", variant="primary")

    nilai_out = gr.Markdown(label="Nilai")
    evaluasi_out = gr.Markdown(label="Evaluasi")

    btn_cari.click(
        cari_materi,
        inputs=[level_in, topik_in, jumlah_pg_in, jumlah_essay_in],
        outputs=[materi_out, state_box, status_out, nilai_out, evaluasi_out, *pg_radios, *essay_boxes],
    )

    btn_submit.click(
        submit_jawaban,
        inputs=[state_box, *pg_radios, *essay_boxes],
        outputs=[state_box, status_out, nilai_out, evaluasi_out, *pg_radios, *essay_boxes],
    )

if __name__ == "__main__":
    demo.launch()

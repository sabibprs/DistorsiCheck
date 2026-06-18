import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import requests
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, hstack
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="DistorsiCheck",
    layout="wide"
)

st.markdown("""
<style>
            
[data-testid="stHeaderActionElements"] {
    display: none !important;
}
            
h1, [data-testid="stCaptionContainer"] {
    text-align: center !important;
}

.stApp {
    background: linear-gradient(180deg, #000000, #434343);
}

header[data-testid="stHeader"] {
    background-color: #000000;
}
/* Input & Textarea */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #535353;
}

/* Tombol */
.stButton > button {
    background-color: #000000;
    border: 1px solid #535353;
    color: white !important;
}

/* Responsif wide layout */
.block-container {
    max-width: 860px !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    margin: auto !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

[data-testid="stExpander"] details {
    background-color: #535353 !important;
}
            
</style>
""", unsafe_allow_html=True)

# ── Lexicon Semantik ─────────────────────────
semantic_lexicon = {
    "hiperbolik": [
        "geger","heboh","gempar","dahsyat","mencekam","mengguncang",
        "menggemparkan","menghebohkan","mengerikan","menakutkan","mengejutkan",
        "bikin kaget","bikin heboh","bikin gempar","bikin geger","bikin merinding",
        "bikin takut","bikin panik","shocking","epik","dramatis",
        "mendadak","tiba-tiba","sontak","panik","kalang kabut","porak poranda",
        "kacau balau","amburadul","kacau","kisruh","ribut","rusuh","ricuh",
        "anarkis","kerusuhan","huru hara","huru-hara","kekacauan","kegaduhan","gaduh",
    ],
    "emosional": [
        "murka","berang","meradang","marah-marah","ngamuk","amuk","meluap",
        "meluap-luap","emosi","naik pitam","panas","memanas","gerah","geram",
        "sangat marah","kemarahan","terancam","menjerit","menangis","menggila",
        "ketakutan","ketakutan massal","panik massal","ancaman serius",
        "bahaya besar","bahaya mengancam","nyawa terancam","terancam punah",
        "waspada tinggi","siaga darurat","pengkhianat","pembohong","penipu",
        "koruptor","penjilat","boneka","badut","tukang bohong","tukang tipu",
        "tukang fitnah","fitnah","munafik","bangsat","brengsek",
    ],
    "provokatif": [
        "gagal total","gagal parah","tidak becus","tidak kompeten","tidak berguna",
        "tidak mampu","habis-habisan","tanpa ampun","paling parah","terburuk",
        "paling buruk","paling jelek","paling bodoh","paling korup","nihil prestasi",
        "nol prestasi","sia-sia","wajib tahu","syok","viral","terbongkar","terungkap",
        "terkuak","terekspos","membongkar","mengungkap","fakta mengejutkan",
        "fakta tersembunyi","fakta mencengangkan","netizen heboh","netizen geram",
        "netizen murka","semua orang","seluruh rakyat","seluruh warga",
        "sepanjang sejarah","sepanjang masa","pertama kali dalam sejarah",
        "direncanakan","rekayasa","skenario","disembunyikan","ditutupi","dimanipulasi",
        "ada yang mengatur","ada yang bermain","dalang","persekongkolan",
        "konspirasi","sabotase","nekat","nekad","biadab","buas","keji",
        "barbarian","tidak berperikemanusiaan",
    ],
    "bias": [
        "katanya","konon","kabarnya","disebut-sebut","diisukan","beredar",
        "diduga kuat","diduga keras","sumber terpercaya","orang dalam",
        "sumber internal","tanpa bukti","tanpa konfirmasi","tanpa verifikasi",
        "tersiar kabar","santer terdengar","berita bohong","berita palsu",
        "fake news","hoaks","hoax","sesat","menyesatkan","disesatkan",
        "ditipu","dibohongi","manipulasi informasi","distorsi informasi",
        "informasi menyesatkan","propaganda","pencucian otak",
    ],
}

def extract_semantic_features(teks):
    teks_lower = str(teks).lower()
    total_kata = max(len(teks_lower.split()), 1)
    skor = {}
    for kategori, kata_list in semantic_lexicon.items():
        kata_urut = sorted(kata_list, key=lambda x: len(x.split()), reverse=True)
        cocok = sum(
            1 for kata in kata_urut
            if re.search(r"\b" + re.escape(kata) + r"\b", teks_lower)
        )
        skor[f"skor_{kategori}"] = round(cocok / total_kata, 4)
    skor["skor_distorsi"] = round(
        skor.get("skor_hiperbolik", 0) * 0.25
        + skor.get("skor_emosional", 0) * 0.25
        + skor.get("skor_provokatif", 0) * 0.30
        + skor.get("skor_bias", 0) * 0.20,
        4,
    )
    return pd.Series(skor)

# ── Preprocessing ────────────────────────────
@st.cache_resource
def load_nlp():
    factory_stop = StopWordRemoverFactory()
    stopwords = set(factory_stop.get_stop_words())

    custom_stopwords = {
        "jakarta", "indonesia", "tahun", "hari", "waktu", "tempat",
        "kata", "ujar", "ungkap", "sebut", "tutur", "ucap",
        "jelas", "tegas", "lanjut", "tambah", "bilang",
        "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
        "januari", "februari", "maret", "april", "mei", "juni",
        "juli", "agustus", "september", "oktober", "november", "desember",
        "wib", "wita", "wit", "pukul", "foto", "video", "detik", "kompas",
        "liputan", "cnn", "antara", "tribun", "okezone"
    }
    stopwords = stopwords.union(custom_stopwords)

    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    return stopwords, stemmer

def preprocess(teks):
    stopwords, stemmer = load_nlp()
    teks = teks.lower()
    teks = re.sub(r"http\S+", " ", teks)
    teks = re.sub(r"www\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = re.sub(r"\b[a-zA-Z]\b", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    tokens = word_tokenize(teks)
    tokens = [w for w in tokens if w not in stopwords]
    teks = stemmer.stem(" ".join(tokens))
    return teks

# ── Load Model ───────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model       = joblib.load(os.path.join(BASE_DIR, "models", "model_distorsi.pkl"))
    tfidf_model = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_model.pkl"))
    return model, tfidf_model

model, tfidf_model = load_model()

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    .sb-label {
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .sb-text {
        font-size: 13px;
        color: #cccccc;
        line-height: 1.7;
        margin-bottom: 20px;
    }
    .sb-desc {
        background-color: #1a2d3d;
        border-left: 3px solid #4a9edd;
        border-radius: 4px;
        padding: 12px 14px;
        margin-top: 8px;
        font-size: 13px;
        color: #7ec8f7;
        line-height: 1.7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:16px; font-weight:700; color:#ffffff; margin-bottom:8px;">
        Informasi Model
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="sb-label">Metode</div>
    <div class="sb-text">
        Sistem ini dibangun menggunakan algoritma Naive Bayes yang dikombinasikan 
        dengan pendekatan analisis semantik untuk mengidentifikasi pola bahasa 
        yang mengindikasikan distorsi informasi.
    </div>

    <div class="sb-label">Kelas</div>
    <div class="sb-text">
        Sistem mengklasifikasikan berita ke dalam 2 kelas yaitu:<br><br>
        <span style="color:#ff6b6b;">■</span> Distorsi<br>
        <span style="color:#6b9fff;">■</span> Normal
    </div>

    <div class="sb-desc">
        Deteksi dilakukan berdasarkan pola bahasa seperti penggunaan kata 
        hiperbolik, emosional, provokatif, dan bias dalam teks berita.
    </div>
    """, unsafe_allow_html=True)

# ── UI ───────────────────────────────────────
st.title("DistorsiCheck")
st.caption("Deteksi Distorsi Informasi Pada Berita Online Menggunakan Metode Naive Bayes dengan Pendekatan Analysis Semantik")
st.markdown("---")

# Input URL
url_input = st.text_input(
    "URL Berita (opsional):",
    placeholder="https://www.detik.com/..."
)

teks_dari_url = ""
if url_input:
    with st.spinner("Mengambil isi berita dari URL..."):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url_input, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            teks_dari_url = " ".join([p.get_text() for p in paragraphs])
            if teks_dari_url.strip():
                st.success("Berhasil mengambil teks dari URL!")
            else:
                st.warning("Teks tidak berhasil diambil, coba tempel manual.")
        except Exception as e:
            st.error("URL tidak valid. Pastikan format URL benar, contoh: https://www.detik.com/...")

# Input teks manual
teks_manual = st.text_area(
    "Teks Berita:",
    value=teks_dari_url,
    height=300,
    placeholder="Tempel judul + isi berita di sini, atau isi otomatis dari URL di atas..."
)

# Tombol prediksi
if st.button("Prediksi", use_container_width=True):
    teks_input = teks_manual.strip()
    if teks_input == "":
        st.warning("Masukkan teks atau URL berita terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            teks_bersih = preprocess(teks_input)

            # Gabungkan TF-IDF + Semantik
            X_tfidf   = tfidf_model.transform([teks_bersih])
            X_sem     = extract_semantic_features(teks_bersih)
            X_sem_csr = normalize(csr_matrix([X_sem.values.astype(float)]))
            X_gabung  = hstack([X_tfidf, X_sem_csr])

            prediksi   = model.predict(X_gabung)[0]
            proba      = model.predict_proba(X_gabung)[0]
            confidence = round(max(proba) * 100, 2)

        st.divider()

        col1, col2 = st.columns([3,1])

        with col1:
            if prediksi == 1:
                st.error("Distorsi Informasi")
            else:
                st.success("Normal")
        with col2:
            st.metric("Persentase", f"{confidence}%")

        with st.expander("Detail Analisis Semantik"):
            skor = extract_semantic_features(teks_input)

            col1, col2 = st.columns(2)

            col1.metric("Hiperbolik", skor["skor_hiperbolik"])
            col2.metric("Emosional", skor["skor_emosional"])

            col1.metric("Provokatif", skor["skor_provokatif"])
            col2.metric("Bias", skor["skor_bias"])

            st.metric("Skor Distorsi Informasi", f"{skor['skor_distorsi']:.4f}")
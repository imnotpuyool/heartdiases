import streamlit as st
import pickle
import numpy as np
import requests
import os

st.set_page_config(
    page_title="CardioCheck — Asesmen Risiko Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


API_URL = os.getenv("API_URL") or st.secrets.get("API_URL", "https://apiheart.up.railway.app")

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        return model, scaler
    return None, None

model, scaler = load_model()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f9f9f9;
    }

    .top-nav {
        background-color: #c0392b;
        padding: 0.85rem 2rem;
        border-radius: 0 0 12px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 12px rgba(192, 57, 43, 0.2);
    }
    .nav-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .nav-brand span {
        color: #fecaca;
        font-weight: 400;
        font-size: 0.85rem;
        margin-left: 0.6rem;
    }
    .nav-tag {
        background: rgba(255,255,255,0.15);
        color: #fff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }

    .page-header {
        margin-bottom: 1.5rem;
    }
    .page-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1a1a1a;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    .page-desc {
        font-size: 0.92rem;
        color: #6b6b6b;
        font-weight: 400;
        line-height: 1.5;
    }

    .stat-strip {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .stat-chip {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-left: 4px solid #c0392b;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        min-width: 130px;
    }
    .stat-chip-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #c0392b;
        line-height: 1;
    }
    .stat-chip-lbl {
        font-size: 0.72rem;
        color: #888;
        font-weight: 500;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .sec-card {
        background: #ffffff;
        border: 1px solid #ebebeb;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .sec-label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #fef2f2;
    }
    .sec-num {
        background: #c0392b;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 800;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: #fafafa !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        color: #1a1a1a !important;
        padding: 0.5rem 0.75rem !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #c0392b !important;
        box-shadow: 0 0 0 3px rgba(192,57,43,0.1) !important;
    }
    label, .stSelectbox label {
        color: #555 !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        background-color: #c0392b !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        transition: background-color 0.2s ease, transform 0.15s ease !important;
        box-shadow: 0 4px 14px rgba(192,57,43,0.25) !important;
    }
    .stButton > button:hover {
        background-color: #a93226 !important;
        transform: translateY(-1px) !important;
    }

    .result-risk {
        background: #fff5f5;
        border: 1px solid #f5c6c6;
        border-left: 5px solid #c0392b;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.2rem;
    }
    .result-safe {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.2rem;
    }
    .result-title-risk {
        font-size: 1.1rem;
        font-weight: 800;
        color: #c0392b;
        margin-bottom: 0.4rem;
    }
    .result-title-safe {
        font-size: 1.1rem;
        font-weight: 800;
        color: #16a34a;
        margin-bottom: 0.4rem;
    }
    .result-desc {
        font-size: 0.88rem;
        color: #555;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .result-meta {
        display: flex;
        gap: 2rem;
        background: #fff;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .rmeta-val {
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1;
        color: #1a1a1a;
    }
    .rmeta-lbl {
        font-size: 0.7rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.25rem;
    }
    .rmeta-rec-lbl {
        font-size: 0.7rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.3rem;
    }
    .rmeta-rec-text {
        font-size: 0.88rem;
        font-weight: 600;
        color: #333;
        line-height: 1.45;
    }
    .badge-risk {
        display: inline-block;
        background: #c0392b;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        margin-left: 0.6rem;
        vertical-align: middle;
    }
    .badge-safe {
        display: inline-block;
        background: #16a34a;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        margin-left: 0.6rem;
        vertical-align: middle;
    }

    .info-note {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.83rem;
        color: #78350f;
        line-height: 1.6;
        margin-top: 1rem;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #ebebeb !important;
    }
    .sb-header {
        background: #c0392b;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .sb-header-title {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 800;
    }
    .sb-header-sub {
        color: #fecaca;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    .sb-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #f3f3f3;
        font-size: 0.85rem;
    }
    .sb-stat-label {
        color: #777;
        font-weight: 500;
    }
    .sb-stat-value {
        color: #c0392b;
        font-weight: 700;
    }
    .step-item {
        display: flex;
        gap: 0.6rem;
        align-items: flex-start;
        margin-bottom: 0.7rem;
        font-size: 0.83rem;
        color: #444;
    }
    .step-num {
        background: #c0392b;
        color: #fff;
        font-size: 0.65rem;
        font-weight: 800;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }

    hr { border-color: #ebebeb !important; }

    .footer {
        text-align: center;
        color: #bbb;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid #ebebeb;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="top-nav">
    <div class="nav-brand"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:28px; height:28px; vertical-align: middle; margin-right: 8px; margin-top: -3px;"><path d="M2 12h3l2-9 3 18 2-13 2 4h6"/></svg>CardioCheck <span>Sistem Asesmen Kesehatan Jantung</span></div>
    <span class="nav-tag">SVM</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">Formulir Asesmen Risiko Penyakit Jantung</div>
    <p class="page-desc">Isi data klinis pasien di bawah ini. Sistem akan menganalisis menggunakan model <b>Support Vector Machine (SVM)</b> untuk memperkirakan risiko penyakit jantung.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-strip">
    <div class="stat-chip">
        <div class="stat-chip-val">97.07%</div>
        <div class="stat-chip-lbl">Akurasi Model</div>
    </div>
    <div class="stat-chip">
        <div class="stat-chip-val">1,025</div>
        <div class="stat-chip-lbl">Data Latih</div>
    </div>
    <div class="stat-chip">
        <div class="stat-chip-val">13</div>
        <div class="stat-chip-lbl">Parameter Klinis</div>
    </div>
    <div class="stat-chip">
        <div class="stat-chip-val">SVM</div>
        <div class="stat-chip-lbl">Algoritma</div>
    </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:44px; height:44px; margin-bottom: 8px;"><path d="M2 12h3l2-9 3 18 2-13 2 4h6"/></svg>
        <div class="sb-header-title">CardioCheck</div>
        <div class="sb-header-sub">Clinical Assessment System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Performa Model**")
    st.markdown("""
    <div class="sb-stat"><span class="sb-stat-label">Akurasi</span><span class="sb-stat-value">97.07%</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Precision</span><span class="sb-stat-value">97%</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Recall</span><span class="sb-stat-value">97%</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Kernel</span><span class="sb-stat-value">RBF (C=5.0)</span></div>
    <div class="sb-stat" style="border:none"><span class="sb-stat-label">Dataset</span><span class="sb-stat-value">1,025 baris</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Panduan Penggunaan**")
    st.markdown("""
    <div class="step-item"><span class="step-num">1</span><span>Isi data profil & tanda vital pasien</span></div>
    <div class="step-item"><span class="step-num">2</span><span>Masukkan hasil laboratorium</span></div>
    <div class="step-item"><span class="step-num">3</span><span>Isi data kardiogram & EKG</span></div>
    <div class="step-item"><span class="step-num">4</span><span>Klik tombol Analisis untuk melihat hasil</span></div>
    """, unsafe_allow_html=True)


with st.form("clinical_form"):

    st.markdown("""
    <div class="sec-label">
        <span class="sec-num">1</span> Profil Demografi &amp; Tanda Vital
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Usia (Tahun)", min_value=18, max_value=120, value=52, step=1)
    with c2:
        sex_label = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        sex = 1 if sex_label == "Laki-laki" else 0
    with c3:
        trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=240, value=125)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-label">
        <span class="sec-num">2</span> Hasil Laboratorium &amp; Metabolisme
    </div>
    """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        chol = st.number_input("Kolesterol Serum (mg/dL)", min_value=100, max_value=600, value=212)
    with c5:
        fbs_label = st.selectbox("Gula Darah Puasa (> 120 mg/dL)", ["Tidak (≤ 120 mg/dL)", "Ya (> 120 mg/dL)"])
        fbs = 1 if "Ya" in fbs_label else 0
    with c6:
        thalach = st.number_input("Detak Jantung Maksimum (BPM)", min_value=60, max_value=230, value=168)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-label">
        <span class="sec-num">3</span> Evaluasi Kardiologi &amp; Elektrokardiogram
    </div>
    """, unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    with c7:
        cp_options = {
            "Nyeri Dada Khas (Typical Angina)": 0,
            "Nyeri Dada Tidak Khas (Atypical Angina)": 1,
            "Nyeri Bukan Angina (Non-anginal Pain)": 2,
            "Tanpa Gejala Nyeri (Asymptomatic)": 3
        }
        cp_selected = st.selectbox("Tipe Nyeri Dada", list(cp_options.keys()))
        cp = cp_options[cp_selected]

        exang_label = st.selectbox("Angina Akibat Olahraga", ["Tidak", "Ya"])
        exang = 1 if exang_label == "Ya" else 0

    with c8:
        restecg_options = {
            "Normal": 0,
            "Abnormalitas Gelombang ST-T": 1,
            "Hipertrofi Ventrikel Kiri": 2
        }
        restecg_selected = st.selectbox("Hasil EKG Istirahat", list(restecg_options.keys()))
        restecg = restecg_options[restecg_selected]

        oldpeak = st.number_input("Depresi ST / Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

    with c9:
        slope_options = {
            "Meningkat (Upsloping)": 0,
            "Datar (Flat)": 1,
            "Menurun (Downsloping)": 2
        }
        slope_selected = st.selectbox("Kemiringan Segmen ST (Slope)", list(slope_options.keys()))
        slope = slope_options[slope_selected]

        ca = st.selectbox("Jumlah Pembuluh Darah Utama (Fluoroskopi)", [0, 1, 2, 3])

    col_thal, _ = st.columns([1, 2])
    with col_thal:
        thal_options = {
            "Normal": 1,
            "Cacat Tetap (Fixed Defect)": 2,
            "Cacat Dapat Pulih (Reversable Defect)": 3
        }
        thal_selected = st.selectbox("Hasil Pemindaian Thalassemia", list(thal_options.keys()))
        thal = thal_options[thal_selected]

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Jalankan Analisis Risiko Kesehatan")


input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

if submitted:
    with st.spinner("Memproses data klinis pasien..."):
        try:
            # First try calling the FastAPI backend on Railway / Local
            prediction = None
            prob_healthy = 0.0
            prob_disease = 0.0

            try:
                clean_url = API_URL.rstrip('/')
                res = requests.post(f"{clean_url}/predict", json={"features": input_features}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    prediction = data.get("prediction")
                    prob_disease = data.get("prob_disease", 0.0)
                    prob_healthy = data.get("prob_healthy", 0.0)
                else:
                    raise Exception(f"API Error HTTP {res.status_code}")
            except Exception as api_err:
                # Fallback to local model if loaded
                if model is not None and scaler is not None:
                    x = np.array(input_features).reshape(1, -1)
                    x_scaled = scaler.transform(x)
                    prediction = int(model.predict(x_scaled)[0])
                    probabilities = model.predict_proba(x_scaled)[0]
                    prob_healthy = float(probabilities[0])
                    prob_disease = float(probabilities[1])
                else:
                    raise Exception(f"Gagal menghubungi FastAPI Backend ({api_err}) dan model lokal tidak tersedia.")

            st.markdown("---")
            st.markdown("#### 📊 Hasil Analisis Prediksi Model")

            if prediction == 1:
                st.markdown(f"""
                <div class="result-risk">
                    <div class="result-title-risk">⚠️ Terindikasi Risiko Penyakit Jantung
                        <span class="badge-risk">Risiko Tinggi</span>
                    </div>
                    <p class="result-desc">
                        Berdasarkan data klinis yang diinput, model SVM memperkirakan pasien memiliki indikasi risiko penyakit jantung.
                        Diperlukan pemeriksaan lanjutan oleh tenaga medis profesional.
                    </p>
                    <div class="result-meta">
                        <div>
                            <div class="rmeta-val" style="color:#c0392b">{prob_disease:.1%}</div>
                            <div class="rmeta-lbl">Probabilitas Risiko</div>
                        </div>
                        <div>
                            <div class="rmeta-rec-lbl">Rekomendasi</div>
                            <div class="rmeta-rec-text">Rujuk segera ke Dokter Spesialis Kardiologi untuk pemeriksaan EKG & Ekokardiografi lanjutan.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-title-safe">✅ Tidak Terindikasi Risiko Signifikan
                        <span class="badge-safe">Risiko Rendah</span>
                    </div>
                    <p class="result-desc">
                        Parameter tanda vital dan laboratorium berada dalam batas aman menurut klasifikasi model.
                        Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.
                    </p>
                    <div class="result-meta">
                        <div>
                            <div class="rmeta-val" style="color:#16a34a">{prob_healthy:.1%}</div>
                            <div class="rmeta-lbl">Probabilitas Sehat</div>
                        </div>
                        <div>
                            <div class="rmeta-rec-lbl">Rekomendasi</div>
                            <div class="rmeta-rec-text">Pertahankan pola hidup sehat & lakukan kontrol rutin berkala ke dokter.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-note">
                ⚠️ <b>Perhatian:</b> Hasil ini bukan merupakan diagnosis klinis resmi.
                Akurasi model <b>97.07%</b> (diuji pada 205 sampel). Nilai probabilitas {prob_disease:.1%} adalah estimasi
                berdasarkan kombinasi parameter pasien ini. Selalu konsultasikan hasil dengan dokter atau tenaga medis.
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Lihat Detail Parameter yang Dianalisis"):
                st.dataframe({
                    "Indikator Medis": [
                        "Usia", "Jenis Kelamin", "Tipe Nyeri Dada (CP)", "Tekanan Darah (trestbps)",
                        "Kolesterol (chol)", "Gula Darah Puasa >120 (fbs)", "EKG Istirahat (restecg)",
                        "Detak Jantung Maks (thalach)", "Angina Olahraga (exang)", "Depresi ST (oldpeak)",
                        "Kemiringan ST (slope)", "Pembuluh Darah (ca)", "Thalassemia (thal)"
                    ],
                    "Nilai Parameter": input_features
                }, use_container_width=True)

        except Exception as e:
            st.error(f"Gagal melakukan analisis data. Error: {str(e)}")

st.markdown("""
<div class="footer">
    CardioCheck Clinical Assessment System &nbsp;·&nbsp; Support Vector Machine (SVM) RBF Kernel &nbsp;·&nbsp; Akurasi 97.07%
</div>
""", unsafe_allow_html=True)
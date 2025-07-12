import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path


BASE_DIR = Path(__file__)
data_path = BASE_DIR / "all_data.csv"
logo_path = BASE_DIR.parent / "images/logo-gundar.png"

# === Load pipeline lengkap & major labels ===
num_pipe  = joblib.load("Models/num_pipe.pkl")
model     = tf.keras.models.load_model("Models/best_student_major_recommendation_model.keras", compile=False)
le        = joblib.load("Models/label_encoder.pkl")
df_major  = pd.read_csv("Datasets/ug_majors_with_passing_grade.csv")

# === Define subjects ===
core_subjects = [
      'agama', 'ppkn', 'bahasa_indonesia', 'matematika', 'bahasa_inggris',
      'seni_budaya', 'penjaskes', 'sejarah'
]
ipa_subjects = [
      'fisika', 'kimia', 'biologi', 'matematika_peminatan', 'informatika',
]
ips_subjects = [
      'ekonomi', 'geografi', 'sosiologi', 'antropologi', 'sastra_indonesia', 'bahasa_asing',
]

all_features  = core_subjects + ipa_subjects + ips_subjects + ['track_bin']
feature_cols = core_subjects + ipa_subjects + ips_subjects

# --- Tambahkan di awal setelah set_page_config ---
st.sidebar.image(str(logo_path), use_container_width=True, caption="Universitas Gunadarma")
st.sidebar.title("Menu")
# --- Inisialisasi session state untuk menyimpan halaman yang dipilih ---
if 'page' not in st.session_state:
      st.session_state.page = "Informasi"  # Halaman default
if st.sidebar.button("Informasi"):
      st.session_state.page = "Informasi"
if st.sidebar.button("Grafik SHAP"):
      st.session_state.page = "Grafik SHAP"
if st.sidebar.button("Prediksi Jurusan"):
      st.session_state.page = "Prediksi Jurusan"
      
# Ambil page terpilih
menu = st.session_state.page

if menu == "Informasi":
      # === Streamlit UI ===
      st.set_page_config(page_title="Re: Best Majors", layout="centered")
      st.title("🎓 Re: Best Majors")
      st.markdown("### Temukan rekomendasi jurusan kuliah terbaik berdasarkan nilai sekolahmu!")
      st.markdown("---")
      
      # st.image("images/logo-gundar.png", use_container_width=True)
      st.markdown("""
      **Re: Best Majors** adalah sistem rekomendasi jurusan kuliah yang dibangun dengan pendekatan *Machine Learning* dan *Neural Network*.
      
      Sistem ini dirancang untuk memudahkan siswa SMA memilih jurusan kuliah yang paling sesuai dengan potensi akademik mereka. Cukup masukkan nilai rapor dan jalur peminatan, lalu sistem yang didukung model Neural Network akan merekomendasikan tiga jurusan terbaik di Universitas Gunadarma untuk Anda. Buat keputusan lebih percaya diri dan mulailah perjalanan akademik Anda dengan tepat!
      
      💡 **Fitur Utama**:
      - Rekomendasi jurusan berbasis nilai akademik sekolah.
      - Analisis SHAP untuk melihat pengaruh nilai pelajaran terhadap hasil prediksi.
      - Gabungan pendekatan berbasis model dan rule-based.

      📌 **Metode**: CRISP-DM (Cross Industry Standard Process for Data Mining)

      🚀 Yuk, mulai prediksi jurusanmu sekarang dari menu **Prediksi Jurusan"** di samping!
      """)
      st.markdown("---")
      st.markdown("Berikut adalah daftar mata pelajaran yang digunakan dalam prediksi sesuai dengan Kurikulum Merdeka terbaru saat ini:")
      
      st.markdown("### Mata Pelajaran Pokok :")
      st.table(pd.DataFrame({"Mata Pelajaran": [s.replace("_", " ").title() for s in core_subjects]}))

      st.markdown("### Peminatan IPA :")
      st.table(pd.DataFrame({"Mata Pelajaran": [s.replace("_", " ").title() for s in ipa_subjects]}))

      st.markdown("### Peminatan IPS :")
      st.table(pd.DataFrame({"Mata Pelajaran": [s.replace("_", " ").title() for s in ips_subjects]}))

      st.markdown("### Jurusan :")
      st.table(pd.DataFrame({
            "Fakultas": [m.title() for m in df_major["faculty"]],
            "Jurusan": [m.title() for m in df_major["major"]],
            }))

      st.markdown("### Mata Pelajaran Terkait:")
      st.table(pd.DataFrame({
            "Jurusan": [m.title() for m in df_major["major"]],
            "Mata Pelajaran": [m.replace(";", ", ").replace("_", " ").title() for m in df_major["related_subjects"]],
            }))

      st.markdown("---")
      
      
elif menu == "Grafik SHAP":
      st.header("📈 SHapley Additive exPlanations (SHAP)")
      
      st.markdown("""
      Ingin tahu mata pelajaran mana yang paling “berkuasa” dalam menentukan jurusan impianmu? __SHapley Additive exPlanations__ Summary Plot kami memetakan seberapa besar kontribusi setiap nilai rapor kalian dari Mata Pelajaran Informatika hingga Seni Budaya dalam mendorong rekomendasi jurusan. Bar paling atas artinya fitur itu kunci: misalnya, nilai Informatika tinggi mengarah ke jurusan IT, nilai Seni Budaya menuntun ke Arsitektur atau Desain Interior.

      Warna unik tiap batang (merah = nilai tinggi, biru = nilai rendah) langsung menunjukkan perilaku model—tanpa teori rumit, kamu bisa “melihat” logika AI kami bekerja! Singkat, visual, dan super intuitif: kini kamu bisa mengeksplorasi kekuatan masing‑masing mata pelajaran sebelum menentukan pilihan jurusan. 🚀
                  
                  """)
      
      # Ganti dengan path gambar SHAP Anda
      try:
            st.image("images/shap_summary_plot.png", caption="SHAP Summary Plot", use_container_width=True)
      except:
            st.warning("Plot SHAP belum tersedia. Silakan tambahkan file `images/shap_summary_plot.png`.")
            
      st.markdown("### Jurusan :")
      st.table(pd.DataFrame({
            "Kode": [m.replace("A", "Class ").title() for m in df_major["code"]],
            "Jurusan": [m.title() for m in df_major["major"]],
            }))
      
      st.markdown("""
      SHAP (SHapley Additive exPlanations) menunjukkan bahwa:
            
      ---
      
      - Seni Budaya (A13: Desain Interior; A11: Arsitektur)
      
      Fitur seni_budaya menempati urutan teratas SHAP bar chart—menandakan bahwa Desain Interior (A13) dan Arsitektur (A11) sangat “sensitif” terhadap nilai Seni Budaya siswa. Dengan nilai tinggi, model secara signifikan lebih condong merekomendasikan kedua jurusan ini, karena keduanya menuntut dasar estetika dan apresiasi seni.
      
      ---
      
      - Informatika (A4: Teknik Informatika; A7: Sistem Informasi; A8: Ilmu Komputer)
      
      Fitur informatika menempati posisi kedua dengan kontribusi luas ke Teknik Informatika (A4), Sistem Informasi (A7), dan Ilmu Komputer (A8). Warna‐warna kelas dominan—terutama Class 3 (A4) dan Class 7 (A8)—memperlihatkan bahwa siswa dengan nilai Informatika bagus sangat mungkin direkomendasikan ke jurusan‐jurusan IT tersebut.
      
      ---
      
      - Bahasa Inggris (A9: Sastra Inggris; A2: Akuntansi; A10: Psikologi)
      
      Fitur bahasa_inggris menjadi faktor penting hampir di seluruh jurusan, namun paling menonjol di Sastra Inggris (A9) dan juga turut memperkuat rekomendasi untuk Akuntansi (A2) serta Psikologi (A10). Ini mencerminkan kebutuhan bahasa yang kuat dalam jurusan humaniora maupun ekonomi-bisnis.
      
      ---
      
      - Matematika & Matematika Peminatan (A3: Teknik Industri; A4: Teknik Informatika; A5: Teknik Mesin; A8: Ilmu Komputer)
      
      Kedua fitur ini (bar biru dan pink-ke­oranye) sangat krusial untuk jurusan‐jurusan teknik dan komputasi. Teknik Industri (A3), Teknik Mesin (A5), dan Ilmu Komputer (A8) menerima lonjakan SHAP pada Matematika, sedangkan Matematika Peminatan khususnya mendorong rekomendasi ke Teknik Informatika (A4) dan Ilmu Komputer (A8).
      
      ---
      
      - Sosiologi & Ekonomi (A1: Manajemen; A2: Akuntansi; A10: Psikologi)
      
      Bar chart menunjukkan sosiologi dan ekonomi sebagai pendorong utama untuk Manajemen (A1) dan Akuntansi (A2) (kelas‐kelas biru tua dan hijau tua), serta untuk Psikologi (A10). Hal ini masuk akal karena ketiga jurusan tersebut memerlukan pemahaman sosial-ekonomi yang kuat.
      
      ---
      
      - Fisika & Kimia (A6: Teknik Elektro; A12: Teknik Sipil; A5: Teknik Mesin)
      
      Nilai fisika memiliki SHAP tertinggi untuk Teknik Elektro (A6) dan Teknik Sipil (A12), sementara kimia menunjukkan puncak kontribusi pada Teknik Industri (A3) dan Teknik Mesin (A5). Ini sesuai dengan kebutuhan dasar sains eksperimental di jurusan‐jurusan tersebut.
      
      ---
      
      - Bahasa Indonesia, Sastra Indonesia & Bahasa Asing (A9: Sastra Inggris; A10: Psikologi; A1/A2: Manajemen/Akuntansi)
      
      Ketiganya berkontribusi moderat—bahasa_indonesia dan sastra_indonesia memengaruhi jurusan humaniora dan sosial (Sastra Inggris, Psikologi), sedangkan bahasa_asing memperkuat pola untuk Sastra Inggris dan juga Psikologi, menunjukkan pentingnya kemampuan komunikasi verbal.
      
      ---
      
      - Geografi & Antropologi (A11: Arsitektur; A13: Desain Interior; A10: Psikologi)
      
      Kedua fitur ini menonjol untuk Arsitektur (A11) dan Desain Interior (A13) (kelas hijau tua dan oren muda), serta berkontribusi untuk Psikologi, mengindikasikan bahwa pemahaman ruang, budaya, dan manusia relevan untuk ketiga jurusan ini.
      
      ---
      
      - PPKn, Biologi, Agama & Penjaskes
      
      ppkn masih menunjukkan beberapa kontribusi ringan untuk Manajemen dan Akuntansi, sedangkan biologi memiliki peran kecil pada Psikologi dan Ilmu Komputer. Sementara itu, agama dan penjaskes praktis berkontribusi nol—model menyimpulkan bahwa dua mata pelajaran terakhir ini tidak signifikan untuk membedakan jurusan.
      """)
      st.markdown("---")
      

# --- Prediksi Jurusan" ---
elif menu == "Prediksi Jurusan":
      st.header("Prediksi Jurusan")
      def predict_major_from_streamlit(input_data, track_bin):
            df_num = pd.DataFrame([{f: float(input_data[f]) for f in feature_cols}]).astype(float)   
            X_num = num_pipe.transform(df_num.values)
            proba = model.predict(X_num)[0]
            idx  = np.argmax(proba)
            code = le.inverse_transform([idx])[0]
            
            # 2) Kumpulkan semua kandidat yang memenuhi passing grade
            candidates = []
            for idx in np.argsort(proba)[::-1]:
                  if proba[idx] <= 0:
                        continue

                  code = le.inverse_transform([idx])[0]
                  row  = df_major.loc[df_major['code'] == code].squeeze()

                  # hitung rata‑rata related_subjects
                  rel    = [s.strip() for s in row.related_subjects.split(';')]
                  scores = [float(input_data.get(s, 0)) for s in rel]
                  if not scores:
                        continue

                  avg = sum(scores) / len(scores)
                  pg  = row.get('passing_grade', 0)
                  if avg >= pg:
                        candidates.append({
                        'code':    code,
                        'faculty': row.faculty,
                        'major':   row.major,
                        'avg':     avg,
                        'p':       proba[idx]
                        })

            # 3) Urutkan berdasarkan avg menurun, ambil top‑3
            top3 = sorted(candidates, key=lambda x: x['avg'], reverse=True)[:3]
            return top3

      track = st.radio("Pilih Jalur Peminatan:", ["IPA","IPS"])
      track_bin = 1 if track=="IPA" else 0
      active_features = core_subjects + (ipa_subjects if track=="IPA" else ips_subjects)
      input_data = {}
      
      st.markdown("---")
      st.subheader("Masukkan Nilai Mata Pelajaran Pokok")
      col1, col2 = st.columns(2)
      for i, feat in enumerate(all_features):
            label = feat.replace("_"," ").title()
            if feat in core_subjects:
                  # pakai text_input untuk bisa dikosongkan
                  with (col1 if i%2==0 else col2):
                        val = st.text_input(
                        label,
                        value="",
                        key=feat,
                        placeholder="0–100"
                        )
                  input_data[feat] = val
            else:
                  input_data[feat] = "0"
                  
      st.markdown("---")
      st.subheader("Masukkan Nilai Mata Pelajaran Peminatan")
      col1, col2 = st.columns(2)
      if track == "IPA":
            track_subjects = ipa_subjects
      else:
            track_subjects = ips_subjects
      for i, feat in enumerate(all_features):
            label = feat.replace("_"," ").title()
            if feat in track_subjects:
                  # pakai text_input untuk bisa dikosongkan
                  with (col1 if i%2==0 else col2):
                        val = st.text_input(
                        label,
                        value="",
                        key=feat,
                        placeholder="0–100"
                        )
                  input_data[feat] = val
                  
      input_data['track_bin'] = track_bin

      st.markdown("---")
      if st.button("Prediksi Jurusan"):
            # 1) Validasi semua active_features terisi
            missing = [f for f in active_features if input_data[f].strip()==""]
            if missing:
                  st.warning(
                        f"Mohon isi nilai untuk semua mata pelajaran: {', '.join([m.replace('_',' ').title() for m in missing])}."
                  )
            else:
                  # Panggil fungsi
                  recs = predict_major_from_streamlit(input_data, track_bin)
                  if not recs:
                        st.warning("Tidak ada jurusan yang memenuhi passing grade.")
                  else:
                        st.header(f"🎯 ({len(recs)}) Rekomendasi Jurusan")
                        for i, r in enumerate(recs,1):
                              st.subheader(f"**{i}. {r['major']}** (Kode: {r['code']})")
                              st.markdown(f"- Fakultas: {r['faculty']}")
                              st.markdown(f"- Rata-rata Nilai Mata Pelajaran Terkait: {r['avg']:.2f}")
                              # tampilkan passing grade di df_major
                              st.markdown(f"- Passing Grade: {df_major[df_major['code'] == r['code']]['passing_grade'].values[0]}")
                              st.markdown("---")
                              
st.caption("📘 Dibuat oleh Ahmad Zaky Humami 2025 – Powered by Machine Learning")
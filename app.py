import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import io, base64
from PIL import Image
import time

BASE_DIR = Path(__file__)
logo_path = BASE_DIR.parent / "images/logo-gundar.png"

@st.cache_resource
def load_resources():
      num_pipe = joblib.load("Models/num_pipe.pkl")
      model     = tf.keras.models.load_model(
            "Models/best_student_major_recommendation_model.keras",
            compile=False
      )
      le        = joblib.load("Models/label_encoder.pkl")
      df_major  = pd.read_csv("Datasets/ug_majors_with_passing_grade.csv")
      return num_pipe, model, le, df_major

num_pipe, model, le, df_major = load_resources()

core_subjects = [
      'agama', 'ppkn', 'bahasa_indonesia', 'matematika', 'bahasa_inggris',
      'seni_budaya', 'penjaskes', 'sejarah'
]
ipa_subjects = [
      'fisika', 'kimia', 'biologi', 'matematika_peminatan', 'informatika'
]
ips_subjects = [
      'ekonomi', 'geografi', 'sosiologi', 'antropologi', 'sastra_indonesia', 'bahasa_asing'
]

all_features  = core_subjects + ipa_subjects + ips_subjects + ['track_bin']
feature_cols = core_subjects + ipa_subjects + ips_subjects

st.set_page_config(
      page_title="Re: Best Majors", 
      layout="wide"
      )

buffer = io.BytesIO()
Image.open(logo_path).save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

st.sidebar.markdown(
      f"""
      <div style="text-align:center;">
            <img src="data:image/png;base64,{b64}" width="250px"/>
      </div>
      """,
      unsafe_allow_html=True
)

st.sidebar.title("Pilih Menu")

if 'page' not in st.session_state:
      st.session_state.page = "Informasi"  # Halaman default
if st.sidebar.button("Informasi", use_container_width=True):
      st.session_state.page = "Informasi"
if st.sidebar.button("Grafik SHAP", use_container_width=True):
      st.session_state.page = "Grafik SHAP"
if st.sidebar.button("Cari Jurusan", use_container_width=True):
      st.session_state.page = "Cari Jurusan"
      
menu = st.session_state.page

if menu == "Informasi":
      st.title("🎓 Re: Best Majors")
      st.markdown("### Temukan rekomendasi jurusan kuliah terbaik berdasarkan nilai sekolahmu!")
      st.markdown("---")
      
      st.markdown("""
      **Re: Best Majors** adalah sistem rekomendasi jurusan kuliah yang dibangun dengan pendekatan *Machine Learning* dan *Neural Network*.
      
      Sistem ini dirancang untuk memudahkan siswa SMA memilih jurusan kuliah yang paling sesuai dengan potensi akademik mereka. Cukup masukkan nilai rapor dan jalur peminatan, lalu sistem yang didukung model Neural Network akan merekomendasikan tiga jurusan terbaik di Universitas Gunadarma untuk Anda. Buat keputusan lebih percaya diri dan mulailah perjalanan akademik Anda dengan tepat!
      

      🚀 Yuk, mulai prediksi jurusanmu sekarang dari menu **Cari Jurusan"** di samping!
      """)
      st.markdown("---")
      st.markdown("""
                  Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi (Kemendikbudristek) secara resmi mengumumkan penerapan Kurikulum Merdeka secara nasional mulai tahun ajaran 2025/2026. Keputusan ini disampaikan oleh Menteri Nadiem Makarim dalam konferensi pers di kantor Kemendikbudristek, Jakarta, pada Rabu (21/5).

                  Kurikulum Merdeka sebelumnya telah diujicobakan sejak 2022 di ribuan sekolah penggerak di seluruh Indonesia. Setelah melalui evaluasi dan penyempurnaan, pemerintah memutuskan untuk menerapkannya di semua jenjang pendidikan dasar dan menengah.
                  "Transformasi pendidikan adalah langkah penting menuju pembelajaran yang berpihak pada murid. Kurikulum Merdeka memberikan keleluasaan bagi sekolah dan guru untuk menyesuaikan materi sesuai konteks dan kebutuhan siswa," ujar Nadiem.
                  Dalam kurikulum ini, siswa diberikan lebih banyak ruang untuk eksplorasi dan pengembangan kompetensi, terutama melalui pembelajaran berbasis proyek. Sementara itu, penilaian berfokus pada proses dan capaian belajar, bukan hanya nilai akhir.
                  Kemendikbudristek juga menyiapkan pelatihan intensif bagi guru dan kepala sekolah guna memastikan transisi berjalan lancar. Sosialisasi dan distribusi modul pembelajaran telah dimulai sejak awal Mei.
                  Penerapan Kurikulum Merdeka diharapkan mampu meningkatkan kualitas pendidikan dan relevansi pembelajaran terhadap tantangan abad ke-21.
                  <a href="https://suarmahasiswaawards.teropongmedia.id/artikel/kemendikbudristek-luncurkan-kurikulum-merdeka-secara-nasional-mulai-tahun-ajaran-2025-2026/#:~:text=Jakarta%2C%2022%20Mei%202025%20—%20Kementerian,mulai%20tahun%20ajaran%202025/2026." target="blank">Suar Mahasiswa Awards
                  </a>
                  """, unsafe_allow_html=True)
      
      st.markdown("""
                  Struktur Kurikulum Merdeka SMA Kelas XI dan XII (Fase F)
                  Pada fase F untuk kelas XI dan XII, struktur mata pelajaran dibagi menjadi lima kelompok utama, yaitu: 

                  **Kelompok Mata Pelajaran Umum**
                  Kelompok mata pelajaran ini wajib diikuti oleh semua siswa SMA.
                  **Kelompok Mata Pelajaran MIPA**
                  Kelompok MIPA terdiri dari ***Matematika, Fisika, Kimia, Biologi, dan Informasi***. Setiap sekolah wajib menyediakan paling sedikit tiga mata pelajaran dalam kelompok ini.
                  **Kelompok Mata Pelajaran IPS**
                  Kelompok IPS terdiri dari ***Ekonomi, Antropologi, Geografi, dan Sosiologi***. Sama seperti kelompok MIPA, setiap sekolah wajib menyediakan paling sedikit tiga mata pelajaran dalam kelompok ini.
                  **Kelompok Mata Pelajaran Bahasa dan Budaya**
                  Kelompok mata pelajaran ini bersifat pilihan. Itu artinya, sekolah bisa memilih untuk membuka kelompok mata pelajaran Bahasa dan Budaya atau tidak sesuai dengan ketersediaan SDM di sekolah.
                  **Kelompok Vokasi dan Prakarya**
                  Kelompok mata pelajaran Vokasi dan Prakarya juga bersifat pilihan. Sekolah bisa mengadakan mata pelajaran ini atau tidak, tergantung dengan ketersediaan SDM di sekolah. 
                  <a href="https://www.quipper.com/id/blog/info-guru/kurikulum-merdeka-sma/" target="blank">Quipper Blog
                  </a>
                  """, unsafe_allow_html=True)
      
      st.markdown("Berikut adalah daftar mata pelajaran yang digunakan dalam prediksi sesuai dengan Kurikulum Merdeka terbaru saat ini:")
      st.markdown("---")
      
      col1, col2, col3 = st.columns(3)
      with col1:
            st.markdown("### Mata Pelajaran Pokok :")
            st.table(pd.DataFrame({
                  "Mata Pelajaran": [s.replace("_", " ").title() for s in core_subjects],
                  }))
      with col2:
            st.markdown("### Peminatan IPA :")
            st.table(pd.DataFrame({
                  "Mata Pelajaran": [s.replace("_", " ").title() for s in ipa_subjects],
                  }))

      with col3:
            st.markdown("### Peminatan IPS :")
            st.table(pd.DataFrame({
                  "Mata Pelajaran": [s.replace("_", " ").title() for s in ips_subjects],
                  }))
            
      st.markdown("---")

      st.markdown("### Jurusan :")
      st.table(pd.DataFrame({
            "Fakultas": [m.title() for m in df_major["faculty"]],
            "Jurusan": [m.title() for m in df_major["major"]],
            "Mata Pelajaran Terkait": [m.replace("_", " ").replace(";", ", ").title() for m in df_major["related_subjects"]],
            }))

      st.markdown("---")
      
elif menu == "Grafik SHAP":
      st.header("📈 SHapley Additive exPlanations (SHAP)")
      
      st.markdown("""
      Ingin tahu mata pelajaran mana yang paling "berkuasa" dalam menentukan jurusan impianmu? __SHapley Additive exPlanations__ Summary Plot kami memetakan seberapa besar kontribusi setiap nilai rapor kalian dari Mata Pelajaran Informatika hingga Seni Budaya dalam mendorong rekomendasi jurusan. Bar paling atas artinya fitur itu kunci: misalnya, nilai Informatika tinggi mengarah ke jurusan IT, nilai Seni Budaya menuntun ke Arsitektur atau Desain Interior.

      Warna unik tiap batang (merah = nilai tinggi, biru = nilai rendah) langsung menunjukkan perilaku model—tanpa teori rumit, kamu bisa "melihat" logika AI kami bekerja! Singkat, visual, dan super intuitif: kini kamu bisa mengeksplorasi kekuatan masing‑masing mata pelajaran sebelum menentukan pilihan jurusan. 🚀
                  """, unsafe_allow_html=True)
      
      st.markdown("---")
      
      try:
            buffer = io.BytesIO()
            Image.open("images/shap_summary_plot_best.png").save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()

            st.markdown(
                  f"""
                  <div style="text-align:center">
                        <img src="data:image/png;base64,{b64}" width="700px" />
                  </div>
                  <p style="text-align:center">
                        SHapley Additive exPlanations 
                  </p>
                  """,
                  unsafe_allow_html=True
            )
      except:
            st.warning("Plot SHAP belum tersedia.")
      
      st.markdown("---")
      st.markdown("""
            SHAP (SHapley Additive exPlanations) menunjukkan bahwa:
      """)
      
      with st.expander("Fisika: Fitur Paling Dominan"):
            st.write('''
            Nilai Fisika menempati urutan tertinggi dalam SHAP summary plot, mengindikasikan bahwa performa pada mata pelajaran ini paling mempengaruhi rekomendasi jurusan—terutama program-program eksakta dan teknik seperti Teknik Fisika (Code 0), Teknik Elektro (Code 6), dan Teknik Sipil (Code 12). Peningkatan satu poin di rapor Fisika secara konsisten "menarik" model ke jurusan yang menuntut pemahaman konsep mekanika, gelombang, dan elektromagnetika.
                  ''')

      with st.expander("Matematika: Dasar Logika dan Kuantitatif"):
            st.write('''
            Sebagai fitur kedua terpenting, Matematika memandu model dalam menilai kecakapan analitis siswa. Jurusan-jurusan seperti Teknik Informatika (Code 4), Teknik Industri (Code 2), dan Ilmu Komputer (Code 8) sangat responsif terhadap nilai Matematika—menunjukkan bahwa kemampuan numerik dan abstraksi algoritmik memiliki bobot kuat dalam pembentukan probabilitas rekomendasi.
                  ''')

      with st.expander("PPKn: Pilar Disiplin dan Kepemimpinan"):
            st.write('''
            Menempati posisi ketiga, PPKn ternyata tidak hanya memengaruhi jurusan IPS dan Hukum, tetapi juga mendorong peluang di Manajemen (Code 1) dan Akuntansi (Code 2). Hal ini mengindikasikan bahwa nilai tinggi pada PPKn—yang mencerminkan kesadaran bernegara dan kedisiplinan—dianggap penting oleh model untuk jurusan yang memerlukan etika profesi dan kepemimpinan.
                  ''')

      with st.expander("Kimia: Gerbang Dunia Industri"):
            st.write('''
            Dengan peringkat keempat, Kimia menjadi penentu kuat bagi jurusan Teknik Kimia (Code 5), Teknik Industri (Code 2), dan Farmasi (Code 16). Siswa yang mencetak nilai Kimia tinggi akan melihat peningkatan eksponensial dalam SHAP value untuk program studi yang bergantung pada reaksi kimia, proses manufaktur, dan formulasi obat.
                  ''')

      with st.expander("Bahasa Inggris: Keterampilan Komunikasi Global"):
            st.write('''
            Bahasa Inggris berada di urutan kelima, memengaruhi rekomendasi tidak hanya pada Sastra Inggris (Code 9) dan Ilmu Komunikasi (Code 14), tetapi juga pada Psikologi (Code 10) dan Manajemen (Code 1). Ini memperlihatkan bahwa kecakapan berbahasa asing—dan kemampuan memahami teks berbahasa Inggris—menjadi aset lintas disiplin dalam memproyeksikan citra akademik siswa.
                  ''')

      with st.expander("Biologi & Bahasa Indonesia: Pilar Life Sciences dan Humaniora"):
            st.write('''
            Nilai Biologi (peringkat keenam) terutama mengangkat peluang di Psikologi (Code 10) dan Farmasi (Code 16), sedangkan Bahasa Indonesia (peringkat ketujuh) mendorong jurusan-jurusan humaniora seperti Sastra Indonesia (Code 15) dan Ilmu Komunikasi (Code 14). Kombinasi kedua mata pelajaran ini menunjukkan dimensi kognitif—mulai dari pemahaman makhluk hidup hingga keahlian retorika.
                  ''')

      with st.expander("Matematika Peminatan & Informatika: Penguat TI"):
            st.write('''
            Menempati peringkat kedelapan dan kesembilan, kedua fitur ini membentuk "sinyal ganda" bagi program-program teknologi: Matematika Peminatan mendukung Teknik Informatika (Code 4) dan Ilmu Komputer (Code 8), sedangkan Informatika (pemrograman) memperkuat rekomendasi untuk Sistem Informasi (Code 7) dan Rekayasa Perangkat Lunak (jika ada).
                  ''')

      with st.expander("Ekonomi & Seni Budaya: Sinergi Bisnis dan Kreativitas"):
            st.write('''
            Ekonomi (peringkat kesepuluh) mengarahkan model ke Manajemen (Code 1), Akuntansi (Code 2), dan Ekonomi Pembangunan (jika tersedia). Sementara Seni Budaya (peringkat kesebelas) tetap relevan untuk Desain Interior (Code 13) dan Arsitektur (Code 11), menegaskan peran kreativitas dan estetika dalam rekomendasi jurusan berbasis seni rupa.
                  ''')

      with st.expander("Sejarah, Sastra Indonesia & Agama: Konteks Sosio-Budaya"):
            st.write('''
            Dalam peringkat 12-14, nilai Sejarah, Sastra Indonesia, dan Agama memberikan "sentuhan akhir" pada rekomendasi jurusan sosial-ilmu seperti Ilmu Komunikasi (Code 14), Sosiologi (Code 17), dan ilmu budaya. Meski dampaknya moderat, fitur ini menambah dimensi pemahaman budaya dan etika ke dalam profil rekomendasi.
                  ''')

      with st.expander("Geografi, Sosiologi, Penjaskes, Bahasa Asing & Antropologi: Kontribusi Terbatas"):
            st.write('''
            Kelima fitur paling bawah (peringkat 15-19) memiliki SHAP value relatif kecil, menandakan bahwa mereka kurang membedakan antar jurusan. Geografi dan Sosiologi sedikit memengaruhi Arsitektur (Code 11) dan perencanaan ruang, sedangkan Penjaskes, Bahasa Asing, dan Antropologi hampir bersifat netral dalam algoritma rekomendasi.
                  ''')
      st.markdown("---")
      

# --- Prediksi Jurusan" ---
elif menu == "Cari Jurusan":
      st.header("Cari Rekomendasi Jurusan")
      def predict_major_from_streamlit(input_data: dict, track_bin: int):
            """
            - input_data: dict mapping feature (nama mapel) -> nilai (float)
            - track_bin : 1 untuk IPA, 0 untuk IPS
            """
            df_num = pd.DataFrame([{f: float(input_data.get(f, 0)) for f in feature_cols}]).astype(float).astype(float)
            X_num = num_pipe.transform(df_num.values)
            proba = model.predict(X_num)[0]
            
            desired = 'IPA' if track_bin == 1 else 'IPS'
            df_track = df_major[df_major['track_type'].str.upper() == desired].copy()

            results = []
            underrated = []

            for _, row in df_track.iterrows():
                  # get passing grade and related subjects
                  pg = row['passing_grade']
                  rels = [s.strip() for s in row['related_subjects'].split(';')]

                  # always include all related subject scores (zeros if missing)
                  vals = [float(input_data.get(subj, 0)) for subj in rels]
                  avg_score = sum(vals) / len(rels) if rels else 0.0

                  # probability for this specific major
                  try:
                        idx_major = le.transform([row['code']])[0]
                        proba_i = float(proba[idx_major])
                  except Exception:
                        proba_i = 0.0

                  entry = {
                        'faculty' : row['faculty'],
                        'major'   : row['major'],
                        'avg'     : avg_score,
                        'pg'      : pg,
                        'proba'   : float(proba_i),
                        'rel_str' : ", ".join(
                        subj.replace('_',' ').title()
                        for subj in sorted(rels, key=lambda x: float(input_data.get(x, 0)), reverse=True)
                        )
                  }

                  if avg_score >= pg:
                        results.append(entry)
                  else:
                        underrated.append(entry)

            results    = sorted(results, key=lambda x: x['proba'], reverse=True)[:3]
            underrated  = sorted(underrated, key=lambda x: x['proba'], reverse=True)[:3]

            return results, underrated

      
      track = st.radio("Pilih Jalur Peminatan:", ["IPA","IPS"])
      track_bin = 1 if track=="IPA" else 0
      
      if 'last_track_bin' not in st.session_state:
            st.session_state.last_track_bin = track_bin

      with st.form("Isi Form"):
            active_features = core_subjects + (ipa_subjects if track=="IPA" else ips_subjects)
            input_data = {}
            
            def predict_with_delay(input_data):
                  with st.spinner("Sedang menghitung nilai anda..."):
                        time.sleep(1)
                  return predict_major_from_streamlit(input_data, track_bin=track_bin)
            
            st.markdown("---")
            st.subheader("Masukkan Nilai Mata Pelajaran Pokok")
            col1, col2 = st.columns(2)
            for i, feat in enumerate(core_subjects):
                  label = feat.replace("_"," ").replace("ppkn","Pendidikan Kewarganegaraan").title()
                  if feat in core_subjects:
                        with (col1 if i % 2 == 0 else col2):
                              val = st.text_input(
                              label,
                              value="",
                              key=feat,
                              placeholder="0.00 - 100.00"
                              )
                        input_data[feat] = val
                  else:
                        input_data[feat] = "0"
                        
            st.markdown("---")
            if st.session_state.last_track_bin != track_bin:
                  with st.spinner(f"Memuat mata pelajaran untuk {track}"):
                        time.sleep(3)
                  st.session_state.last_track_bin = track_bin
                  
            st.subheader("Masukkan Nilai Mata Pelajaran Peminatan")
            col1, col2 = st.columns(2)
            if track == "IPA":
                  track_subjects = ipa_subjects
            else:
                  track_subjects = ips_subjects
                  
            for i, feat in enumerate(track_subjects):
                  label = feat.replace("_"," ").title()
                  if feat in track_subjects:
                        with (col1 if i % 2 == 0 else col2):
                              val = st.text_input(
                              label,
                              key=feat,
                              placeholder="0.00 - 100.00"
                              )
                        input_data[feat] = val
                  else:
                        input_data[feat] = "0"
                  
            input_data['track_bin'] = track_bin
      
            st.markdown("---")
            if st.form_submit_button("Mulai Cari", use_container_width=True):
                  
                  missing = [f for f in active_features if input_data[f].strip()==""]
                  if missing:
                        st.warning(
                        f"Mohon isi nilai untuk semua mata pelajaran: "
                        f"**{', '.join([m.replace('_',' ').title() for m in missing])}.**"
                        )
                        st.markdown("---")
                  else:
                        recs, nrecs = predict_with_delay(input_data)
                        if recs:
                              st.markdown("---")
                              st.success(f"#### 🎉 Prediksi berhasil! 🎯 {len(recs)} Jurusan yang cocok buat kamu!")
                              cols = st.columns(len(recs))
                              for col, r in zip(cols, recs):
                                    with col:
                                          st.markdown(f"### {r['major']}")
                                          st.markdown(f"- **Fakultas:** {r['faculty']}")
                                          st.markdown(f"- **Passing Grade:** {r['pg']:.2f}")
                                          st.markdown("##### Rata-rata Nilai Terkait")
                                          st.markdown(f"{r['rel_str']} : **{r['avg']:.2f}**")
                              st.markdown("---")
                        else:
                              st.markdown("---")
                              st.warning("Maaf sepertinya tidak ada jurusan yang memenuhi Passing Grade.")
                              cols = st.columns(len(nrecs))
                              for col, r in zip(cols, nrecs):
                                    with col:
                                          st.markdown(f"### {r['major']}")
                                          st.markdown(f"- **Fakultas:** {r['faculty']}")
                                          st.markdown(f"- **Passing Grade:** {r['pg']:.2f}")
                                          st.markdown("##### Rata-rata Nilai Terkait")
                                          st.markdown(f"{r['rel_str']} : **{r['avg']:.2f}**")
                              st.markdown("---")
      st.markdown("---")
                        
st.caption("📘 Dibuat oleh Ahmad Zaky Humami | 50422138 | S1 Informatika | Universitas Gunadarma - 2025")
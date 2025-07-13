import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import io, base64
from PIL import Image

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
      'fisika', 'kimia', 'biologi', 'matematika_peminatan', 'informatika'
]
ips_subjects = [
      'ekonomi', 'geografi', 'sosiologi', 'antropologi', 'sastra_indonesia', 'bahasa_asing'
]

all_features  = core_subjects + ipa_subjects + ips_subjects + ['track_bin']
feature_cols = core_subjects + ipa_subjects + ips_subjects

# === Streamlit UI ===
st.set_page_config(
      page_title="Re: Best Majors", 
      layout="wide"
      )

buffer = io.BytesIO()
Image.open(logo_path).save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

# langsung inject HTML
st.sidebar.markdown(
      f"""
      <div style="text-align:center">
            <img src="data:image/png;base64,{b64}" width="250px" caption="Universitas Gunadarma"/>
      </div>
      <p style="text-align:center">
            Universitas Gunadarma
      </p>
      """,
      unsafe_allow_html=True
)

st.sidebar.title("Menu")

st.markdown(
      """
      <style>
            /* perlu target kontainer utama Streamlit */
            div[data-testid="stAppViewContainer"] > div {
                  padding: 50px 0px !important;
            }

            /* --- Override Sidebar --- */
            /* 1) Paksa section sidebar jadi fixed width */
            section[data-testid="stSidebar"] {
            min-width: 500px;
            max-width: 500px;
            }

            /* 2) Pastikan inner block mengikuti */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            max-width: 450px;
            width: 450px;
            }
            
      </style>
      """,
      unsafe_allow_html=True,
)

# --- Inisialisasi session state untuk menyimpan halaman yang dipilih ---
if 'page' not in st.session_state:
      st.session_state.page = "Informasi"  # Halaman default
if st.sidebar.button("Informasi", use_container_width=True):
      st.session_state.page = "Informasi"
if st.sidebar.button("Grafik SHAP", use_container_width=True):
      st.session_state.page = "Grafik SHAP"
if st.sidebar.button("Cari Jurusan", use_container_width=True):
      st.session_state.page = "Cari Jurusan"
      
# Ambil page terpilih
menu = st.session_state.page

if menu == "Informasi":
      st.title("🎓 Re: Best Majors")
      st.markdown("### Temukan rekomendasi jurusan kuliah terbaik berdasarkan nilai sekolahmu!")
      st.markdown("---")
      
      # st.image("images/logo-gundar.png", use_container_width=True)
      st.markdown("""
      **Re: Best Majors** adalah sistem rekomendasi jurusan kuliah yang dibangun dengan pendekatan *Machine Learning* dan *Neural Network*.
      
      Sistem ini dirancang untuk memudahkan siswa SMA memilih jurusan kuliah yang paling sesuai dengan potensi akademik mereka. Cukup masukkan nilai rapor dan jalur peminatan, lalu sistem yang didukung model Neural Network akan merekomendasikan tiga jurusan terbaik di Universitas Gunadarma untuk Anda. Buat keputusan lebih percaya diri dan mulailah perjalanan akademik Anda dengan tepat!
      

      🚀 Yuk, mulai prediksi jurusanmu sekarang dari menu **Cari Jurusan"** di samping!
      """)
      st.markdown("---")
      st.markdown("""
                  Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi (Kemendikbudristek) secara resmi mengumumkan penerapan Kurikulum Merdeka secara nasional mulai tahun ajaran 2025/2026. Keputusan ini disampaikan oleh Menteri Nadiem Makarim dalam konferensi pers di kantor Kemendikbudristek, Jakarta, pada Rabu (21/5).

                  Kurikulum Merdeka sebelumnya telah diujicobakan sejak 2022 di ribuan sekolah penggerak di seluruh Indonesia. Setelah melalui evaluasi dan penyempurnaan, pemerintah memutuskan untuk menerapkannya di semua jenjang pendidikan dasar dan menengah.
                  “Transformasi pendidikan adalah langkah penting menuju pembelajaran yang berpihak pada murid. Kurikulum Merdeka memberikan keleluasaan bagi sekolah dan guru untuk menyesuaikan materi sesuai konteks dan kebutuhan siswa,” ujar Nadiem.
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
      st.set_page_config(
      page_title="Re: Best Majors", 
      layout="wide"
      )
      
      st.markdown(
            """
            <style>
            /* perlu target kontainer utama Streamlit */
            div[data-testid="stAppViewContainer"] > div {
                  padding: 50px 500px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
      )

      st.header("📈 SHapley Additive exPlanations (SHAP)")
      
      st.markdown("""
      Ingin tahu mata pelajaran mana yang paling “berkuasa” dalam menentukan jurusan impianmu? __SHapley Additive exPlanations__ Summary Plot kami memetakan seberapa besar kontribusi setiap nilai rapor kalian dari Mata Pelajaran Informatika hingga Seni Budaya dalam mendorong rekomendasi jurusan. Bar paling atas artinya fitur itu kunci: misalnya, nilai Informatika tinggi mengarah ke jurusan IT, nilai Seni Budaya menuntun ke Arsitektur atau Desain Interior.

      Warna unik tiap batang (merah = nilai tinggi, biru = nilai rendah) langsung menunjukkan perilaku model—tanpa teori rumit, kamu bisa “melihat” logika AI kami bekerja! Singkat, visual, dan super intuitif: kini kamu bisa mengeksplorasi kekuatan masing‑masing mata pelajaran sebelum menentukan pilihan jurusan. 🚀
                  """, unsafe_allow_html=True)
      
      st.markdown("---")
      
      # Ganti dengan path gambar SHAP Anda
      try:
            buffer = io.BytesIO()
            Image.open("images/shap_summary_plot_best.png").save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()

            # langsung inject HTML
            st.markdown(
                  f"""
                  <div style="text-align:center">
                        <img src="data:image/png;base64,{b64}" width="700px" caption="Universitas Gunadarma"/>
                  </div>
                  <p style="text-align:center">
                        Universitas Gunadarma
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
      
      with st.expander("Seni Budaya: Fondasi Kreativitas Juara"):
            st.write('''
            Seni Budaya menempati puncak grafik SHAP—fitur ini paling berpengaruh terhadap rekomendasi jurusan Desain Interior (A13) dan Arsitektur (A11). Bayangkan: setiap poin tambahan dalam nilai Seni Budaya langsung “mengangkat” peluang siswa ke program-program yang menuntut kepekaan visual dan imajinasi tinggi. Artinya, investasi pada seni bukan sekadar menghias portofolio, tapi benar‑benar “menggerakkan jarum” model kami dalam memetakan passion Anda!
                  ''')

      with st.expander("Matematika: Mesin Utama Rekomendasi Teknik"):
            st.write('''
            Di posisi kedua, Matematika menunjukkan betapa krusialnya logika dan numerik dalam memengaruhi jurusan teknik—dari Teknik Industri (A3), Teknik Informatika (A4), hingga Ilmu Komputer (A8). Setiap angka di buku rapor Anda “berbicara” ke model: semakin tinggi kemampuan kuantitatif, semakin besar bobotnya dalam memosisikan Anda ke jurusan‐jurusan berbasis angka dan algoritma.
                  ''')

      with st.expander("Bahasa Inggris: Pintu Gerbang Humaniora & Bisnis"):
            st.write('''
            Menjadi sorotan ketiga, Bahasa Inggris tak hanya penting untuk jurusan sastra, tetapi juga Akuntansi (A2) dan Psikologi (A10). Nilai bagus di sini memperlihatkan keterampilan komunikasi global—sesuatu yang diminati semua fakultas. Dengan kemampuan bahasa yang memukau, Anda “didorong” model ke program yang menuntut presentasi, riset, dan analisis teks kompleks.
                  ''')

      with st.expander("Informatika: Sinyal Kuat ke Dunia IT"):
            st.write('''
            Fitur Informatika tercatat urutan keempat—penanda jelas bahwa kecakapan coding dan logika komputasi menjadi “magnet” utama bagi Teknik Informatika (A4), Sistem Informasi (A7), dan Ilmu Komputer (A8). Saat Anda mencetak nilai tinggi di mapel ini, model kami otomatis “menyala” untuk jalur‐jalur teknologi paling mutakhir di kampus.
                  ''')

      with st.expander("Sosiologi: Pendorong Program Sosial-Ilmiah"):
            st.write('''
            Peringkat kelima milik Sosiologi: siswa dengan nilai unggul di sini akan melihat lonjakan SHAP terhadap jurusan Manajemen (A1), Akuntansi (A2), dan Psikologi (A10). Ini membuktikan bahwa kemampuan memahami dinamika masyarakat dan perilaku manusia menjadi aset penting dalam ranah sosial-ekonomi dan psikologis.
                  ''')

      with st.expander("Bahasa Indonesia & Sastra Indonesia: Pilar Humaniora"):
            st.write('''
            Di peringkat keenam dan ketujuh, keduanya memperkuat rekomendasi ke jurusan–jurusan humaniora seperti Sastra Inggris (A9) dan Psikologi (A10). Nilai kuat di mapel kebahasaan ini menandakan ketajaman analisis teks dan kefasihan berargumen—kemampuan krusial untuk riset kualitatif dan penulisan ilmiah.
                  ''')

      with st.expander("Fisika & Matematika Peminatan: Kunci Dunia Eksakta"):
            st.write('''
            ‘Fisika’ (peringkat 8) dan ‘Matematika Peminatan’ (peringkat 9) menegaskan: siswa dengan pijakan sains eksperimental dan matematika tingkat lanjut semakin condong ke Teknik Elektro (A6), Teknik Sipil (A12), dan Ilmu Komputer (A8). Dua fitur ini bekerja sinergis—membentuk “jembatan” antara teori abstrak dan aplikasi teknik.
                  ''')

      with st.expander("Ekonomi & Geografi: Membaca Peta Bisnis & Ruang"):
            st.write('''
            Masuk ke posisi 10 dan 11, Ekonomi memicu rekomendasi untuk Manajemen dan Akuntansi, sementara Geografi memperkuat peluang di Arsitektur (A11), Desain Interior (A13), dan Psikologi (A10). Ini menunjukkan bahwa wawasan ekonomi‐spasial serta pemahaman ruang-budaya punya peran penting di jurusan bisnis dan perencanaan.
                  ''')

      with st.expander("Kimia, Bahasa Asing, PPKn & Antropologi: Penguat Spesialisasi"):
            st.write('''
            Kimia (12) menyediakan dasar bagi Teknik Industri (A3) dan Teknik Mesin (A5), sedangkan Bahasa Asing (13) mendukung program humaniora. PPKn (14) dan Antropologi (15) memberi kontribusi moderat pada jurusan sosial-ekonomi. Meski lebih “halus,” fitur‐fitur ini menyempurnakan “fit” rekomendasi kami—memberikan warna khusus di profil setiap calon mahasiswa.
                  ''')

      with st.expander("Biologi, Penjaskes, Agama & Sejarah: Kontribusi Minimal"):
            st.write('''
            Terakhir, nilai Biologi, Penjaskes, Agama, dan Sejarah (peringkat 16–19) memiliki dampak yang sangat kecil. Model menyimpulkan bahwa keempat mapel ini kurang membedakan antara program studi, sehingga cenderung “menetral” dalam rekomendasi.
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
            # 1) Siapkan DataFrame numerik & hitung probabilitas
            df_num = pd.DataFrame(
                  [{f: float(input_data[f]) for f in feature_cols}]
            ).astype(float)
            X_num = num_pipe.transform(df_num.values)
            proba = model.predict(X_num)[0]  # array shape (n_classes,)

            # 2) Filter df_major sesuai track_bin
            #    track_type bisa 'IPA', 'IPS', atau 'IPA/IPS'
            desired = 'IPA' if track_bin == 1 else 'IPS'
            df_track = df_major[
                  df_major['track_type'].str.upper().isin([desired, 'IPA/IPS'])
            ].copy()

            # 3) Hitung rata‑rata related_subjects dan kumpulkan yang lulus passing_grade
            results = []
            for _, row in df_track.iterrows():
                  code = row['code']
                  pg   = row['passing_grade']
                  rels = [s.strip() for s in row['related_subjects'].split(';')]

                  # ambil nilai, kalau ada 0 → skip seluruh jurusan
                  vals = []
                  for subj in rels:
                        v = float(input_data.get(subj, 0))
                        if v <= 0:
                              vals = []
                              break
                        vals.append(v)
                  if not vals:
                        continue

                  avg_score = sum(vals) / len(vals)
                  if avg_score >= pg:
                        # simpan
                        results.append({
                        'code'   : code,
                        'faculty': row['faculty'],
                        'major'  : row['major'],
                        'avg'    : avg_score,
                        'pg'     : pg,
                        # format daftar mapel terurut desc
                        'rel_str': ", ".join(
                              subj.replace('_',' ').title()
                              for subj, _ in sorted(zip(rels, vals),
                                                      key=lambda x: x[1], reverse=True)
                        )
                        })

            # 4) Pilih top 3 berdasar avg (menurun)
            top3 = sorted(results, key=lambda x: x['avg'], reverse=True)[:3]
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
                        placeholder="0.00 - 100.00"
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
            if feat in track_subjects==ipa_subjects:
                  # pakai text_input untuk bisa dikosongkan
                  with (col1 if i%2==0 else col2):
                        val = st.text_input(
                        label,
                        value="",
                        key=feat,
                        placeholder="0.00 - 100.00"
                        )
                  input_data[feat] = val
            if feat in track_subjects==ips_subjects:
                  # pakai text_input untuk bisa dikosongkan
                  with (col1 if i%2==1 else col2):
                        val = st.text_input(
                        label,
                        value="",
                        key=feat,
                        placeholder="0.00 - 100.00"
                        )
                  input_data[feat] = val
                  

      input_data['track_bin'] = track_bin

      st.markdown("---")
      if st.button("Mulai Cari", use_container_width=True):
            missing = [f for f in active_features if input_data[f].strip()==""]
            if missing:
                  st.warning(
                  f"Mohon isi nilai untuk semua mata pelajaran: "
                  f"**{', '.join([m.replace('_',' ').title() for m in missing])}.**"
                  )
                  st.markdown("---")

            else:
                  recs = predict_major_from_streamlit(input_data, track_bin)
                  if not recs:
                        st.warning("Maaf sepertinya tidak ada jurusan yang memenuhi Passing Grade.")
                  else:
                        st.success(f"### 🎉 Prediksi berhasil! 🎯 {len(recs)} Jurusan yang cocok buat kamu!")
                        # st.header(f"🎯 {len(recs)} Rekomendasi Jurusan")
                        cols = st.columns(len(recs))
                        for col, r in zip(cols, recs):
                              with col:
                                    st.markdown(f"### {r['major']}  \nKode: **{r['code']}**")
                                    st.markdown(f"- **Fakultas:** {r['faculty']}")
                                    st.markdown(f"- **Passing Grade:** {df_major.loc[df_major['code']==r['code'],'passing_grade'].values[0]:.2f}")
                                    st.markdown("##### Rata‑rata Nilai Mapel Terkait")
                                    st.markdown(f"{r['rel_str']} : **{r['avg']:.2f}**")
                        
                        st.markdown("---")
                        
st.caption("📘 Dibuat oleh Ahmad Zaky Humami | [50422138] | S1 Informatika | Universitas Gunadarma - 2025")
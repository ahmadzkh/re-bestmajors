# Proyek Penulisan Ilmiah

**FAKULTAS TEKNOLOGI INDUSTRI
PRODI INFORMATIKA
UNIVERSITAS GUNADARMA 2025**

- **Nama      :** Ahmad Zaky Humami
- **NPM       :** 50422138

<p align="center">
    <img width="250" alt="logo-gundar" src="https://github.com/user-attachments/assets/e737e330-00c4-4688-a706-25a8de016f63" />

</p>

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-orange)](https://github.com/ahmadzkh/re-bestmajors)  
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)]()

## Project Overview
Pemilihan jurusan kuliah yang tepat merupakan langkah awal krusial bagi keberhasilan akademik dan karier masa depan siswa. Dalam konteks pendidikan tinggi, setiap mahasiswa sebaiknya belajar sesuai bidang keahlian yang cocok dengan minat, bakat, dan kemampuan mereka, agar mampu menghasilkan sumber daya manusia berkualitas sesuai bidang keahliannya. Namun pada kenyataannya banyak calon mahasiswa mengalami kebingungan dalam memilih jurusan. Akibatnya, terjadi ketidaksesuaian antara latar belakang kemampuan akademik siswa SMA dengan program studi yang diambil di perguruan tinggi. Ketidaksesuaian ini sering menyebabkan mahasiswa keluar dari perkuliahan atau pindah jurusan di tengah studi. Misalnya, sebuah laporan Kemendikbudristek tahun 2020 mencatat sebanyak 601.333 mahasiswa mengalami putus kuliah (drop out) pada tahun tersebut [(Nabilah Nur Alifah, 2022)](https://goodstats.id/article/jurusan-kuliah-dengan-mahasiswa-do-terbanyak-2021-2Z8VD#:~:text=Istilah%20drop%20out%20atau%20DO,mahasiswa%20putus%20kuliah%20pada%202020). Angka ini sebagian dipicu oleh kesalahan dalam memilih jurusan, di mana siswa kurang memahami potensi akademik dan karakteristik tiap jurusan. Penelitian menunjukkan bahwa kesalahan pemilihan jurusan berdampak negatif pada prestasi akademik dan kesejahteraan psikologis mahasiswa. 

Berbagai upaya telah dikembangkan untuk membantu siswa menentukan jurusan yang sesuai. Contohnya, platform pendidikan seperti Youthmanual telah menganalisis lebih dari 400.000 data profil siswa untuk menemukan minat dan memberikan panduan pemilihan program studi yang tepat. Upaya tersebut mengurangi kebingungan siswa dalam pemilihan jurusan dan menurunkan risiko drop out. Selain itu, penelitian-penelitian sebelumnya telah mengusulkan pengembangan sistem rekomendasi jurusan berbasis data akademik siswa. Misalnya, sistem rekomendasi jurusan menggunakan metode C4.5 telah dicoba untuk calon mahasiswa baru, dan metode Naïve Bayes juga pernah diterapkan untuk tujuan serupa. Hasil-hasil ini menunjukkan bahwa pendekatan data mining dapat membantu proses pemilihan jurusan secara lebih terstruktur dan objektif. 

Berdasarkan latar belakang tersebut, penelitian ini bertujuan mengembangkan sebuah sistem rekomendasi jurusan yang memanfaatkan nilai akademik siswa SMA sebagai input. Sistem ini dibangun dengan menggunakan model Neural Network berbasis Keras Sequential yang mampu mempelajari pola hubungan kompleks antara nilai-nilai akademik siswa dan jurusan yang diambil mahasiswa. Data akademik dan jurusan yang digunakan merupakan dataset siswa SMA dan mahasiswa dari Universitas Gunadarma. Pengembangan sistem dilakukan dengan pendekatan CRISP-DM (Cross-Industry Standard Process for Data Mining), sehingga proyek ini mengikuti tahap-tahap standar industri data mining mulai dari pemahaman bisnis hingga penerapan (deployment). Antarmuka pengguna sistem diimplementasikan sebagai aplikasi web interaktif menggunakan framework Streamlit, guna memudahkan siswa SMA mengakses rekomendasi jurusan secara mudah dan cepat. Dengan demikian, sistem ini diharapkan menjadi solusi bagi siswa dalam menentukan jurusan kuliah yang sesuai dengan potensi akademik mereka, sekaligus menambah wawasan stakeholders mengenai pemanfaatan data mining dalam pendidikan.

Penelitian ini dibatasi pada hal-hal berikut:
- Dataset yang digunakan terdiri dari data nilai akademik siswa SMA dan data jurusan yang ditempuh oleh mahasiswa Universitas Gunadarma sebagai target rekomendasi.
- Sistem difokuskan untuk merekomendasikan jurusan perguruan tinggi (S1) berdasarkan potensi akademik siswa, bukan merekomendasikan spesialisasi atau mata kuliah tertentu.
- Model analisis yang digunakan adalah Neural Network sederhana (Keras Sequential) tanpa membandingkan performa dengan algoritma lain.
- Proses pengembangan mengikuti kerangka kerja CRISP-DM, yang meliputi fase: pemahaman bisnis, pemahaman data, persiapan data, pemodelan, evaluasi, dan deployment.
- Implementasi sistem dilakukan sebagai aplikasi web menggunakan Streamlit, tanpa membahas integrasi ke platform lain atau pengujian lapangan yang luas.
- Penelitian berfokus pada aspek pengembangan dan evaluasi model rekomendasi, dan tidak mencakup evaluasi jangka panjang terhadap efektivitas rekomendasi dalam pengambilan keputusan siswa sesungguhnya.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah sebagai berikut:

1. Bagaimana merancang dan mengimplementasikan sistem rekomendasi jurusan kuliah berbasis nilai akademik siswa SMA?
2. Bagaimana membangun model prediksi jurusan menggunakan Neural Network (Keras Sequential) berdasarkan data nilai akademik dan riwayat jurusan mahasiswa Universitas Gunadarma?
3. Bagaimana menerapkan pendekatan CRISP-DM dalam tahapan perancangan sistem rekomendasi ini?
4. Bagaimana mengintegrasikan model prediksi ke dalam aplikasi web yang dibangun menggunakan framework Streamlit untuk memfasilitasi pengguna dalam memperoleh rekomendasi jurusan?

### Goals
Tujuan dari penelitian ini adalah:
1. Merancang sebuah sistem rekomendasi jurusan kuliah bagi siswa SMA berdasarkan nilai akademik dan jurusan sebelumnya, guna membantu proses pemilihan jurusan yang sesuai dengan potensi akademik.
2. Mengembangkan model Neural Network (menggunakan Keras Sequential) untuk memprediksi jurusan yang tepat berdasarkan data nilai akademik siswa.
3. Menerapkan pendekatan CRISP-DM sebagai metodologi pengembangan untuk menjamin bahwa tahapan analisis data dan pembuatan model dilakukan secara sistematis.
4. Membangun prototipe aplikasi web menggunakan Streamlit untuk menyajikan hasil rekomendasi secara interaktif kepada pengguna (siswa, guru, konselor).

### Solution
Untuk menyelesaikan permasalahan rekomendasi jurusan ini, saya menggunakan satu pendekatan pemodelan, yaitu jaringan saraf tiruan (Neural Network) dengan Keras Sequential. Model ini dibangun sebagai rangkaian lapisan (layers) yang saling berurutan: dimulai dari lapisan masukan (input layer) yang menerima nilai-nilai akademik siswa sebagai fitur, diikuti oleh beberapa lapisan tersembunyi (hidden layers) dengan fungsi aktivasi ReLU untuk menangkap hubungan non‑linier antar mata pelajaran, dan diakhiri oleh lapisan keluaran (output layer) beraktivasi softmax yang menghasilkan probabilitas untuk setiap kode jurusan Universitas Gunadarma.

Semenjak fase CRISP‑DM "Modeling", arsitektur dan jumlah neuron pada tiap lapisan disesuaikan melalui eksperimen hyperparameter (jumlah unit, laju dropout, optimizer, dan batch size) untuk memaksimalkan akurasi prediksi. Model dilatih menggunakan algoritma Adam pada loss fungsi categorical_crossentropy, kemudian dievaluasi dengan metrik akurasi dan confusion matrix. Dengan pendekatan ini, jaringan saraf mampu mempelajari pola kompleks nilai rapor siswa dan memproyeksikannya ke rekomendasi tiga jurusan terbaik secara andal.


## Data Understanding
### Sumber Data
Dataset Nilai Akademik Siswa (__student_grades_5000.csv__)
Dataset ini disusun secara sintetis dengan bantuan model GPT untuk mensimulasikan kondisi nyata nilai rapor siswa SMA yang akan mendaftar ke Universitas Gunadarma. Nilai-nilai mata pelajaran pada dataset dibuat mengikuti skala 10-98, sesuai rentang penilaian Kurikulum Merdeka terbaru yang diterapkan oleh Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia. Setiap baris mewakili satu siswa, mencakup delapan mata pelajaran pokok (agama, PPKn, Bahasa Indonesia, Matematika, Bahasa Inggris, Seni Budaya, Penjaskes, Sejarah) serta mata pelajaran peminatan IPA atau IPS. Penciptaan data ini bertujuan memberikan distribusi nilai yang lebih variatif dan representatif, sekaligus mendekati pola sebaran akademik siswa dalam dunia nyata.

Dataset Jurusan Universitas Gunadarma (__ug_majors.csv__)
File ini memuat daftar 13 program studi di Universitas Gunadarma, lengkap dengan kode jurusan, nama fakultas, dan mata pelajaran terkait yang menjadi dasar rekomendasi. Selain itu, setiap jurusan dilengkapi kolom passing grade, yakni nilai ambang rata‑rata mata pelajaran terkait yang merefleksikan batas minimal kelayakan pendaftaran. Nilai passing grade diperoleh dari riset data historis dan kebijakan penerimaan di Universitas Gunadarma, lalu disisipkan secara eksplisit ke dalam dataset. Dengan demikian, dataset ini siap dipakai untuk memadukan rekomendasi model Neural Network dan penerapan aturan ambang kelulusan (threshold), sehingga menghasilkan sistem rekomendasi yang tidak hanya akurat secara prediksi, tapi juga realistis dalam konteks kebijakan seleksi perguruan tinggi.

### Deskripsi Variabel
#### Dataset Student Grades
| Variabel               | Keterangan                                                         |
| ---------------------- | ------------------------------------------------------------------ |
| `student_id`           | ID unik setiap siswa, format "S00001" hingga "S05000"              |
| `track`                | Jalur peminatan siswa: "IPA" atau "IPS"                            |
| **Core Subjects**      |                                                                    |
| `agama`                | Nilai mata pelajaran Pendidikan Agama                              |
| `ppkn`                 | Nilai mata pelajaran PPKn (Pendidikan Pancasila & Kewarganegaraan) |
| `bahasa_indonesia`     | Nilai mata pelajaran Bahasa Indonesia                              |
| `matematika`           | Nilai mata pelajaran Matematika                                    |
| `bahasa_inggris`       | Nilai mata pelajaran Bahasa Inggris                                |
| `seni_budaya`          | Nilai mata pelajaran Seni & Budaya                                 |
| `penjaskes`            | Nilai mata pelajaran Pendidikan Jasmani, Olahraga, dan Kesehatan   |
| `sejarah`              | Nilai mata pelajaran Sejarah Indonesia & Dunia                     |
| **Elective IPA**       | (kolom diisi jika `track` = "IPA", else NaN)                       |
| `fisika`               | Nilai mata pelajaran Fisika                                        |
| `kimia`                | Nilai mata pelajaran Kimia                                         |
| `biologi`              | Nilai mata pelajaran Biologi                                       |
| `matematika_peminatan` | Nilai Matematika peminatan (khusus IPA)                            |
| `informatika`          | Nilai mata pelajaran Informatika                                   |
| **Elective IPS**       | (kolom diisi jika `track` = "IPS", else NaN)                       |
| `ekonomi`              | Nilai mata pelajaran Ekonomi                                       |
| `sosiologi`            | Nilai mata pelajaran Sosiologi                                     |
| `geografi`             | Nilai mata pelajaran Geografi                                      |
| `antropologi`          | Nilai mata pelajaran Antropologi                                   |
| `sastra_indonesia`     | Nilai mata pelajaran Sastra Indonesia                              |
| `bahasa_asing`         | Nilai mata pelajaran Bahasa Asing (Inggris/Mandarin dll.)          |


```
Info Dataset Student Grades : 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4902 entries, 0 to 4901
Data columns (total 21 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   student_id            4902 non-null   object 
 1   track                 4902 non-null   object 
 2   agama                 4902 non-null   float64
 3   ppkn                  4902 non-null   float64
 4   bahasa_indonesia      4902 non-null   float64
 5   matematika            4902 non-null   float64
 6   bahasa_inggris        4902 non-null   float64
 7   seni_budaya           4902 non-null   float64
 8   penjaskes             4902 non-null   float64
 9   sejarah               4902 non-null   float64
 10  fisika                2775 non-null   float64
 11  kimia                 2775 non-null   float64
 12  biologi               2775 non-null   float64
 13  matematika_peminatan  2775 non-null   float64
 14  informatika           2775 non-null   float64
 15  ekonomi               2127 non-null   float64
 16  sosiologi             2127 non-null   float64
 17  geografi              2127 non-null   float64
 18  antropologi           2127 non-null   float64
 19  sastra_indonesia      2127 non-null   float64
 20  bahasa_asing          2127 non-null   float64
dtypes: float64(19), object(2)
memory usage: 804.4+ KB
```
##### Student Grades Missing Values
Variabel | Missing Value
----------|----------
Student ID | 0
Track  | 0
Agama  | 0
Ppkn  | 0
Bahasa Indonesia  | 0
Matematika  | 0
Bahasa Inggris  | 0
Seni Budaya  | 0
Penjaskes  | 0
Sejarah  | 0
Fisika  | 2127
Kimia  | 2127
Biologi  | 2127
Matematika Peminatan  | 2127
Informatika  | 2127
Ekonomi  | 2775
Sosiologi  | 2775
Geografi  | 2775
Antropologi  | 2775
Sastra Indonesia  | 2775
Bahasa Asing  | 2775

Disini saya membiarkan saja missing values pada dataset student grades, dikaranakan pada siswa dengan track "IPA" pasti tidak memiliki nilai pada mata pelajaran peminatan "IPS" dan begitu juga sebaliknya.

##### Student Grades Duplicated
```
Jumlah Duplikasi Data Student Grades : 0
```

#### Dataset Majors
| Variabel          | Keterangan                                                                          |
| ----------------- | ----------------------------------------------------------------------------------- |
| code              | Kode unik jurusan di Universitas Gunadarma (misalnya A1, A2, ..., A13)              |
| major             | Nama jurusan (misalnya: Teknik Informatika, Akuntansi, Psikologi, dsb)              |
| faculty           | Fakultas tempat jurusan berada (misalnya: Fakultas Teknik, Ekonomi, dll)            |
| track\_type       | Jenis peminatan siswa yang cocok (IPA / IPS)                                        |
| related\_subjects | Daftar mata pelajaran yang relevan untuk jurusan ini (misalnya: Matematika, Fisika) |


```
Info Dataset Majors : 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13 entries, 0 to 12
Data columns (total 5 columns):
 #   Column            Non-Null Count  Dtype 
---  ------            --------------  ----- 
 0   code              13 non-null     object
 1   major             13 non-null     object
 2   faculty           13 non-null     object
 3   track_type        13 non-null     object
 4   related_subjects  13 non-null     object
dtypes: object(5)
memory usage: 648.0+ bytes
```
##### Majors Missing Values
Variabel | Missing Value
----------|----------
Code | 0
Major  | 0
Faculty  | 0
Track Type | 0
Related Subjects | 0

Tidak terdapat Missing Value pada dataset Majors.

##### Majors Duplicated
```
Jumlah Duplikasi Data Majors : 0
```

### Exploratory Data Analysis (EDA)
#### Proporsi Jalur Peminatan (track)

<p align="center">
    <img width="989" height="590" alt="track_distribution" src="https://github.com/user-attachments/assets/a795582b-8d9f-43f2-aa02-4cb8de1a00c2" />
</p>

#### Distribusi Nilai Mata Pelajaran Pokok

<p align="center">
    <img width="1189" height="788" alt="core_subjects" src="https://github.com/user-attachments/assets/9565ab14-1fec-45f0-a0f6-329fc5f678e6" />
</p>

#### Distribusi Nilai Mata Pelajaran IPA

<p align="center">
    <img width="1189" height="567" alt="ipa_subjects" src="https://github.com/user-attachments/assets/e63d8ea9-f418-41fc-97ab-7540eb1c916a" />
</p>

##### Distribusi Nilai Mata Pelajaran IPS

<p align="center">
    <img width="1189" height="567" alt="ips_subjects" src="https://github.com/user-attachments/assets/82b11231-f40e-4bd7-b3ba-07e80453819e" />
</p>

Distribusi Nilai Mata Pelajaran IPS:


## Data Preparation
### Feature Enginering

#### Threshold Passing Grade
Pertama saya melakukan Feature Engineering untuk perhitungan passing grade pada fase Data Preparation untuk memastikan rekomendasi jurusan tidak sekadar berdasar peringkat skor saja, tetapi juga merefleksikan ambang kompetisi yang riil. Dengan fungsi `add_passing_grade()`, saya menghitung persentil ke‑75 dari distribusi rata‑rata nilai siswa pada mata pelajaran terkait untuk setiap jurusan. Persentil ini saya ambil sebagai threshold—artinya, saya menganggap seorang calon siswa "memenuhi syarat" jika rata‑rata nilai pada mata pelajaran jurusan tersebut berada di atas nilai yang dicapai 75% siswa dalam dataset sintetis.

Secara teknis, saya mem‑parse kolom `related_subjects` menjadi daftar nama mapel (lowercase, underscore), lalu memilih kolom nilai siswa yang cocok. Setelah menghitung rata‑rata setiap siswa untuk daftar mapel itu, saya mengambil nilai persentil ke‑75 sebagai passing_grade. Hasilnya, setiap baris pada df_major kini memuat kolom passing_grade nilai ambang minimal yang kemudian saya pakai di fungsi prediksi untuk memfilter jurusan sebelum model Neural Network memberikan rekomendasi akhir.

```
def add_passing_grade(df_students: pd.DataFrame,
                        df_majors: pd.DataFrame,
                        pct: float = 75.0,
                        related_col: str = 'related_subjects') -> pd.Series:
      """
      Hitung persentil rata-rata nilai siswa untuk setiap major,
      lalu kembalikan Series passing_grade.
      """
      thresholds = []
      for _, mj in df_majors.iterrows():
            rel = [s.strip().lower().replace(' ', '_')
                  for s in mj[related_col].split(';')]
            
            cols = [c for c in df_students.columns if c in rel]
            
            if not cols:
                  thresholds.append(np.nan)
                  continue
            
            avg_scores = df_students[cols].mean(axis=1)
            value = np.percentile(avg_scores.dropna(), pct)
            thresholds.append(round(value, 2))
            
      return pd.Series(thresholds, name='passing_grade')

df_major['passing_grade'] = add_passing_grade(df_student, df_major, pct=75)
```
Output :

<p align="center">
    <img width="1512" height="853" alt="passing_grades" src="https://github.com/user-attachments/assets/09763a94-0df5-4481-a5e2-d1229a3a51e5" />
</p>

#### Recommended Major
Lanjut saya melakukan perhitungan skor potensi siswa untuk setiap jurusan berdasarkan nilai mata pelajaran terkait. Fungsi compute_major_score() akan menerima satu baris data siswa dan daftar mata pelajaran yang relevan untuk sebuah jurusan, lalu mengembalikan rata‑rata nilai dari mata pelajaran tersebut.

Setelah itu, untuk setiap siswa, kita melakukan iterasi ke seluruh daftar jurusan (df_major) dan hanya mempertimbangkan jurusan yang sesuai dengan jalur (IPA, IPS, atau IPA/IPS). Skor jurusan dihitung dengan memanggil compute_major_score(), kemudian jurusan dengan skor rata‑rata tertinggi dipilih sebagai rekomendasi rule‑based awal. Hasil akhir disimpan ke dalam kolom recommended_major pada df_student:
```
def compute_major_score(student_row: pd.Series, major_subjects: list[str]) -> float:
      """Return mean score of the given related_subjects for one student."""
      vals = [student_row[subj] 
                  for subj in major_subjects 
                  if pd.notna(student_row.get(subj))]
      return np.mean(vals) if vals else np.nan

recommended_codes = []
for _, student in df_student.iterrows():
      track = student['track'].upper()
      best_score = -np.inf
      best_code  = None

      df_track = df_major[df_major['track_type'] == track]

      for _, mj in df_track.iterrows():
            subjects = [s.strip() for s in mj['related_subjects'].split(';')]
            
            score = compute_major_score(student, subjects)
            
            pg = mj['passing_grade']
            if not np.isnan(score) and score >= pg:
                  if score > best_score:
                        best_score = score
                        best_code  = mj['code']
      
      recommended_codes.append(best_code)

df_student['recommended_major'] = recommended_codes
```
Output Distribusi Recommended Major :

<p align="center">
    <img width="200" alt="recommended_major" src="https://github.com/user-attachments/assets/03d71c94-b531-4101-b5f0-6f6b2431533d" />
</p>


### Mapping Track
Lanjut saya menambahkan kolom `track` di `df_student` untuk mencerminkan peminatan IPA atau IPS siswa. Pada fase Data Preparation, saya melakukan one‑hot encoding sederhana dengan mengubah nilai string "IPA" menjadi 1 dan "IPS" menjadi 0, lalu menyimpannya di kolom track_bin. Langkah ini mempermudah model Neural Network membedakan jalur peminatan tanpa perlu memproses teks secara langsung. Dengan demikian, setiap baris data siswa kini memiliki fitur numerik track_bin yang digunakan bersama nilai mata pelajaran untuk melatih dan melakukan inferensi pada model.
```
df_student['track_bin'] = df_student['track'].map({'IPA': 1, 'IPS': 0}).astype(int)
final_df = df_student.drop(columns=['student_id','track'])
```
Ouput : 

<p align="center">
    <img width="1895" height="336" alt="mapping_track" src="https://github.com/user-attachments/assets/ad619c3a-69d0-48cb-94a5-9fd1d7def777" />
</p>


### Label Encoder
Pada tahap Label Encoding, saya mengonversi target rekomendasi jurusan (__recommended_major__) yang awalnya berupa kode string seperti "A1", "A2", …, "A13" menjadi format numerik sehingga dapat ditangani oleh model. Saya menggunakan LabelEncoder dari scikit‑learn untuk memetakan setiap kode jurusan ke indeks bilangan bulat unik. Setelah proses encoding, variabel target (y) berwujud array integer, sementara daftar classes menyimpan urutan kode jurusan asli. Langkah ini memastikan kompatibilitas dengan fungsi predict() dan metrik evaluasi pada model Neural Network Keras Sequential, sekaligus mempertahankan kemampuan untuk melakukan inverse transform saat menampilkan kembali kode jurusan hasil prediksi.
```
le = LabelEncoder().fit(final_df['recommended_major'])
classes = list(le.classes_)
```
Output:
```
['A1',
 'A10',
 'A11',
 'A12',
 'A13',
 'A14',
 'A15',
 'A16',
 'A17',
 'A18',
 'A19',
 'A2',
 'A20',
 'A21',
 'A3',
 'A4',
 'A5',
 'A6',
 'A7',
 'A8',
 'A9',
 None]
```

### Numerical Features & Split Data
Saya memilih 19 fitur numerik—terdiri dari delapan mata pelajaran pokok dan sebelas mata pelajaran peminatan—sebagai variabel input utama. Setelah mengumpulkan data siswa dan target rekomendasi jurusan, saya melakukan train-validation-test split secara stratified (70% train, 15% validation, 15% test) berdasarkan label jurusan. Pendekatan stratified ini menjaga distribusi setiap jurusan tetap konsisten di ketiga subset, sehingga evaluasi akurasi dan metrik lain mencerminkan performa model pada populasi yang representatif.
```
feature_cols = core_subjects + ipa_subjects + ips_subjects

X = final_df[feature_cols]
y_idx = le.transform(final_df['recommended_major'])
```
```
X_train, X_temp, y_train, y_temp = train_test_split(
      X, y_idx, train_size=0.70, stratify=y_idx, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
```
Output :
```
Train: 3431 samples
Validation: 735 samples
Test: 736 samples
```

### Simple Imputer & Standard Scaler
Untuk menangani missing values—karena perbedaan jalur IPA/IPS yang membuat beberapa kolom "kosong"—saya menggunakan SimpleImputer(strategy='mean') untuk mengganti nilai NaN dengan rata‑rata kolom, diikuti dengan StandardScaler() untuk melakukan normalisasi z‑score (mean = 0, std = 1). Kedua langkah ini saya bungkus dalam Pipeline sehingga preprocessing dapat diterapkan konsisten pada data train, validation, dan test, serta pada data inference baru.
```
num_pipe = Pipeline([
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler',  StandardScaler())
])

X_train_scaled = num_pipe.fit_transform(X_train)
X_test_scaled  = num_pipe.transform(X_test)
X_val_scaled = num_pipe.transform(X_val)
```

## Model Development

### Neural Network Keras Sequential
Saya merancang model Keras Sequential dengan konfigurasi layer berikut. Input Layer yang menerima vektor 19 fitur hasil preprocessing. Dense Layer 1: 64 neuron, aktivasi ReLU. Dropout: rate 0.3, untuk mengurangi overfitting. Dense Layer 2: 32 neuron, aktivasi ReLU. Dropout: rate 0.2. Output Layer: jumlah neuron sesuai panjang dari kelas jurusan, dan aktivasi softmax.

```
model = Sequential([
      Input(shape=(X_train_scaled.shape[1],)),
      Dense(64, activation='relu'),
      Dropout(0.3),
      Dense(32, activation='relu'),
      Dropout(0.2),
      Dense(len(classes), activation='softmax'),
])

model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
)

model.summary()

callbacks = EarlyStopping(
      monitor='val_loss',
      patience=5,
      restore_best_weights=True
)

history = model.fit(
      X_train_scaled, y_train,
      validation_split=0.15,
      validation_data=(X_val_scaled, y_val),
      epochs=100,
      batch_size=32,
      callbacks=[callbacks],
      verbose=1
)
```

<p align="center">
    <img width="607" height="505" alt="sequential_layers" src="https://github.com/user-attachments/assets/1de3b069-d046-4e42-a7df-c374ff910cc3" />
</p>

Model ini saya kompilasi dengan optimizer Adam, loss categorical_crossentropy, dan metrik accuracy. Selain itu, saya menambahkan callback EarlyStopping (monitor='val_loss", patience=5) agar pelatihan berhenti otomatis saat tidak ada peningkatan, meminimalkan overfitting dan menghemat waktu komputasi.

### SHAP
SHAP (SHapley Additive exPlanations) Summary Plot menampilkan kontribusi rata‑rata absolut setiap fitur terhadap prediksi model, diukur sebagai nilai SHAP. Sumbu horizontal menggambarkan "mean(|SHAP value|)": semakin panjang batang, semakin besar peran fitur tersebut dalam memengaruhi keputusan model. Fitur-fitur diurutkan dari yang paling penting (puncak grafik) hingga yang paling tidak berpengaruh (bawah grafik).

<p align="center">
    <img width="790" height="900" alt="shap_summary_plot" src="https://github.com/user-attachments/assets/d837b6b9-e9a5-4211-b4ac-006a0b725f91" />
</p>

Warna pada setiap batang mewakili nilai aktual fitur: titik‑titik merah menandakan sampel dengan nilai fitur tinggi, biru menandakan sampel dengan nilai rendah. Misalnya, pada grafik di atas fitur informatika dan seni_budaya berada di urutan teratas, menunjukkan bahwa variasi nilai kedua mata pelajaran ini paling banyak menggeser probabilitas rekomendasi jurusan; siswa dengan nilai informatika tinggi lebih cenderung dipetakan ke jurusan IT, sedangkan nilai seni_budaya tinggi mendorong rekomendasi jurusan-jurusan berbasis estetika (Desain Interior, Arsitektur). Dengan demikian, SHAP memberikan gambaran kuantitatif sekaligus intuitif tentang mana mata pelajaran yang paling krusial bagi model dalam menentukan jurusan terbaik bagi setiap siswa.


### LIME
LIME (Local Interpretable Model‑agnostic Explanations) LIME membantu kita memahami "mengapa" model Neural Network mengambil keputusan tertentu untuk tiap sampel siswa. Daripada melihat keseluruhan dataset, LIME membuat model lokal sederhana (biasanya regresi linier) di sekitar titik data yang ingin dijelaskan. Pada grafik LIME:

<p align="center">
    <img width="1061" height="262" alt="lime_model" src="https://github.com/user-attachments/assets/ee84bf70-0479-4881-84da-7120d48391ec" />
</p>

1. Panel kiri memperlihatkan probabilitas prediksi masing‑masing jurusan, sehingga Anda tahu seberapa yakin model.

2. Diagram batang di tengah (bar chart) menampilkan lima fitur (mata pelajaran) teratas yang paling banyak mendorong prediksi ke satu jurusan—misalnya A2 (Sistem Informasi). Batang ke kanan berarti fitur tersebut "menambah" probabilitas jurusan; batang ke kiri berarti "mengurangi".

3. Tabel di kanan memuat nilai Z‑score (baku) fitur pada sampel: semakin positif nilainya (mis. ppkn 1.26), semakin besar dorongan ke prediksi, dan sebaliknya.

Dengan LIME, kita mendapatkan penjelasan yang lokal dan intuitif: untuk setiap siswa, kita dapat melihat mata pelajaran mana yang menjadi "bintang" pendorong rekomendasi model, serta mana yang justru menahan probabilitasnya. Penjelasan ini memudahkan siswa maupun orang tua memahami dan percaya pada rekomendasi jurusan yang dihasilkan.

## Tuning Model
### RandomizedSearchCV
Saya menerapkan __RandomizedSearchCV__ untuk menyetel hyperparameter Neural Network Keras saya secara efisien dan terukur. Alih‑alih mencoba setiap kombinasi parameter secara exhaustif yang bisa memakan waktu puluhan jam tetapi dengan __RandomizedSearchCV__ secara acak mengambil ratusan konfigurasi dari ruang pencarian yang telah saya tentukan (misalnya rentang learning rate, dropout rate, jumlah unit lapisan), lalu mengevaluasi masing‑masing melalui cross‑validation. Dengan cara ini, model diuji pada beberapa "lipatan" data yang berbeda, sehingga saya mendapatkan gambaran stabil tentang performanya. Setelah seluruh iterasi selesai, __RandomizedSearchCV__ memilih kombinasi dengan skor validasi tertinggi sebagai hyperparameter final. Pendekatan acak ini tidak hanya jauh lebih cepat daripada grid search konvensional, tetapi juga mampu menjelajahi area hyperparameter yang mungkin luput dari grid search yang kaku.

#### Define Parameter
```
keras_clf = KerasClassifier(
      model       = build_model,
      validation_split = 0.15,
      epochs      = 100,
      batch_size  = 32,
      callbacks   = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
      verbose     = 1,
)

param_dist = {
      "model__optimizer":      ["adam", "rmsprop"],
      "model__learning_rate":  [1e-2, 1e-3, 5e-4],
      "model__activation":     ["relu", "elu", "gelu", "tanh"],
      "model__dropout_rate":   [0.2, 0.3, 0.4],
      "model__first_units":    [64, 128],
      "model__second_units":   [32, 64],
      "batch_size":            [16, 32]
}
```
#### Train Hyperparameter
```
best_params_overall = None
best_score = 0

random_search = RandomizedSearchCV(
      estimator = keras_clf,
      param_distributions = param_dist,
      n_iter = 200,
      cv = 3,
      scoring = "accuracy",
      random_state = 42,
      return_train_score = True,
      verbose = 1,
      n_jobs = -1,
)

random_search.fit(X_train_scaled, y_train)

best_score = random_search.best_score_
best_params_overall = random_search.best_params_
```

<p align="center">
    <img width="790" height="900" alt="shap_summary_plot_best" src="https://github.com/user-attachments/assets/042d0a47-eb40-44ce-9c3a-93310149e7a5" />
</p>

<p align="center">
    <img width="1073" height="265" alt="lime_best_model" src="https://github.com/user-attachments/assets/bccb2919-4584-4d8a-8622-11086c2936a8" />
</p>

## Evaluasi
### Evaluasi Model Keras Sequential (Sebelum Tuning)
#### Accuracy per Epoch

<p align="center">
    <img width="567" height="455" alt="epoch_accuracy" src="https://github.com/user-attachments/assets/7b3fac17-c3e2-4c78-beab-d62ced7627c4" />
</p>

Grafik "Accuracy per Epoch" memperlihatkan bahwa pada fase awal pelatihan (epoch 0-5), akurasi training dan validasi sama‑sama melonjak cepat—training accuracy dari sekitar 30 % ke 55 % dan validation accuracy dari 33 % ke 60 %. Setelah epoch ke‑10, laju peningkatan akurasi mulai melambat, namun validation accuracy terus menunjukkan tren naik yang stabil hingga mencapai puncaknya di kisaran 82-84 % pada akhir epoch (65). Sementara itu, training accuracy tercatat stabil di rentang 73-75 %, sedikit lebih rendah dari validation accuracy, menandakan model tidak mengalami overfitting signifikan. Kurva yang relatif "bergelombang kecil" pada kedua metrik menggambarkan adanya proses optimasi gradien yang konsisten—cukup agresif untuk menangkap pola tanpa membuat model terlalu kaku pada data training.

#### Loss per Epoch

<p align="center">
    <img width="567" height="455" alt="epoch_loss" src="https://github.com/user-attachments/assets/ba7475f6-6f04-43f6-a9cb-c673fd64fea7" />
</p>

Pada grafik "Loss per Epoch", loss training dan validation keduanya menurun tajam di epoch 0-10, dari nilai awal sekitar 2,7 (training) dan 2,3 (validation) turun ke kisaran masing‑masing 1,2 dan 0,8. Setelah itu, loss training terus menurun secara perlahan dan merata hingga mencapai sekitar 0,75 pada epoch akhir, sedangkan validation loss menurun lebih cepat dan stabil di kisaran 0,53-0,60. Jarak yang relatif konsisten antara dua kurva—dengan validation loss selalu sedikit di bawah training loss—mengonfirmasi bahwa model tidak hanya semakin fit terhadap data training, tetapi juga generalisasi terhadap data baru terus membaik. Tidak terlihat tanda‑tanda divergensi atau kenaikan loss validation, sehingga bisa disimpulkan bahwa jumlah epoch dan parameter regularisasi (dropout, learning rate) telah dipilih secara tepat untuk mencegah overfitting.

#### Confussion Matrix
### Evaluasi Model Keras Sequential Terbaik (Sebelum Tuning)

<p align="center">
    <img width="501" height="682" alt="class_report_model" src="https://github.com/user-attachments/assets/e7171d2b-d414-4064-bf89-f0e2c75906aa" />
    <img width="900" height="790" alt="cm_model" src="https://github.com/user-attachments/assets/6475718f-e157-467c-893c-c1fb71501b8d" />
</p>

Confusion matrix awal menunjukkan performa yang cukup baik pada mayoritas kelas dengan diagonal yang tebal, misalnya kelas Code 22 berhasil mengklasifikasikan 229 dari 231 sampel dengan benar, kelas Code 4 mencapai 40/42, dan kelas Code 12 mencapai 32/34. Rata‑rata akurasi per kelas berada di kisaran 85-99%. Meski demikian, sejumlah kelas berjumlah data terbatas—seperti Code 3 (21/23 benar), Code 8 (20/22 benar) dan Code 16 (27/29 benar)—masih menunjukkan mis‑classifikasi 1-2 sampel, dan beberapa pola off‑diagonal konsisten terlihat pada pertukaran antar Code 1, Code 6, dan Code 18. Hal ini mengindikasikan bahwa meski model telah mempelajari pola fitur akademik dengan baik, ia masih kesulitan membedakan jurusan yang overlap fiturnya tinggi atau yang memiliki sedikit contoh.

### Evaluasi Model Keras Sequential Terbaik (Setelah Tuning)

<p align="center">
    <img width="504" height="682" alt="class_report_best" src="https://github.com/user-attachments/assets/3b4b9e29-ae86-4b18-baf1-597b419248fb" />
    <img width="900" height="790" alt="cm_best_model" src="https://github.com/user-attachments/assets/a3c08032-ed7d-4851-bf22-d1181474f9d3" />
</p>

Pada confusion matrix model terbaik, hampir semua kelas menyentuh peningkatan akurasi: Code 4 naik menjadi 41/42, Code 3 menjadi 22/23, Code 8 menjadi 21/22, Code 10 menjadi 18/19, Code 12 menjadi 34/35, Code 16 menjadi 29/30, dan Code 19 menjadi 31/32 sampel yang terprediksi benar. Kelas besar seperti Code 22 juga tetap stabil di 221/223. Selain itu, jumlah mis‑label global menurun dan distribusi kesalahan menjadi lebih tersebar—pertukaran antar Code 1-2 dan Code 6-7 kini hanya terjadi pada 1-2 sampel saja. Hal ini menunjukkan bahwa tuning hyperparameter dengan RandomizedSearchCV telah berhasil meningkatkan stabilitas dan konsistensi model, meski kelas dengan sangat sedikit data (seperti Code 15 dan Code 17) masih berpotensi mendapat manfaat dari data tambahan.

### Kesimpulan
Berdasarkan evaluasi performa model menggunakan confusion matrix, terbukti bahwa penerapan RandomizedSearchCV pada arsitektur Keras Sequential berhasil meningkatkan akurasi klasifikasi jurusan secara signifikan. Model terbaik menunjukkan peningkatan konsistensi pada diagonal confusion matrix, dengan rata‑rata akurasi per kelas yang naik dari kisaran 85–99 % menjadi 90–100 % dan distribusi kesalahan off‑diagonal yang lebih merata. Hal ini menegaskan bahwa kombinasi pemilihan jumlah layer, fungsi aktivasi, learning rate, dan dropout rate yang optimal mampu memetakan pola distribusi nilai akademik siswa ke dalam jurusan target di Universitas Gunadarma secara andal.

Meskipun demikian, masih terdapat residual error pada jurusan dengan jumlah sampel terbatas (misalnya Code 15 dan Code 17) dan beberapa kasus overlap fitur antar jurusan serupa. Temuan ini mengindikasikan bahwa kualitas dan kuantitas data akademik semata belum sepenuhnya memadai untuk menghasilkan rekomendasi yang benar‑benar personal dan bebas bias, khususnya pada program studi yang under‑represented atau memiliki karakteristik nilai yang mirip.

### Saran
Sebagai pengembangan selanjutnya, penulis merekomendasikan:

- Perluasan Data Non‑Akademik: Mengintegrasikan variabel tambahan seperti skor minat‑bakat, hasil asesmen kepribadian, partisipasi ekstrakurikuler, dan latar belakang sosial‑ekonomi guna memperkaya fitur model. Pendekatan ini diharapkan dapat mengurangi overlap distribusi nilai akademik murni dan menambah konteks personalisasi rekomendasi.

- Augmentasi dan Pengayaan Sampel: Menambah volume data pada jurusan yang jumlah sampelnya sedikit melalui pengumpulan data sukarela atau teknik augmentasi sintetis. Strategi ini penting untuk meminimalkan bias model terhadap kelas dominan dan meningkatkan generalisasi pada kelas minoritas.

- Perluasan Cakupan Jurusan: Memperluas ruang lingkup program studi yang menjadi target rekomendasi, tidak hanya lintas fakultas di Universitas Gunadarma tetapi juga institusi mitra lain. Hal ini akan memperkaya opsi pilihan bagi calon mahasiswa dan meningkatkan relevansi sistem.

- Eksperimen Multimodal dan Ensemble: Menyelidiki integrasi data teks—seperti esai motivasi atau rekomendasi guru—serta penerapan ensemble learning dengan mengombinasikan beberapa model klasifikasi, untuk meningkatkan ketahanan sistem terhadap variasi data dan memperoleh prediksi yang lebih robust.

- Dengan implementasi saran‑saran di atas, diharapkan sistem rekomendasi pemilihan jurusan berbasis machine learning ini dapat beralih dari sekadar akurat secara statistik menjadi lebih adaptif, kontekstual, dan memberikan nilai tambah nyata bagi calon mahasiswa.

## Reference
- Badan Pusat Statistik. 2022. Statistik Pendidikan 2022. Badan Pusat Statistik. Jakarta.

- Muttaqin et al. 2023. "Pengenalan Data Mining". Yayasan Kita Menulis, Yogyakarta.

- Purwati, Neni, Rini Nurlistiani dan Oscar Devinsen. 2020. "Data Mining dengan Algoritma Neural Network dan Visualisasi Data untuk Prediksi Kelulusan Mahasiswa". Jurnal Informatika, 20(2), hlm. 156-163.

- Raihan, Muhammad Iqbal. 2024. "Implementasi Backpropagation untuk Rekomendasi Jurusan Peminatan Mahasiswa Program Studi Teknik Informatika di Universitas Islam Balitar". Jurnal Ilmiah Sistem Informasi dan Teknik Informatika (JISTI), 7(2), hlm. 219-226.

- Rianti, Afika, Nuur Wachid Abdul Majid dan Ahmad Fauzi. 2023. "CRISP-DM: Metodologi Proyek Data Science". Prosiding Seminar Nasional Teknologi Informasi dan Bisnis (SENATIB) 2023.

- Siardizal dan Rosy Dasmita. 2016. "Jaringan Syaraf Tiruan untuk Menentukan Jurusan di Sekolah Menengah Atas (SMA) (Studi Kasus: SMA Negeri 1 Sungai Penuh)". Jurnal Ilmu Pengetahuan dan Sistem Informasi, 2: 91-100.

- Ulfa, Annisa, Doni Winarso dan Edo Arribe. 2020. "Sistem Rekomendasi Jurusan Kuliah Bagi Calon Mahasiswa Baru Menggunakan Algoritma C4.5". Jurnal FASILKOM, 10(1): 61-65.

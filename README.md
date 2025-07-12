# Proyek Penulisan Ilmiah

**FAKULTAS TEKNOLOGI INDUSTRI
PRODI INFORMATIKA
UNIVERSITAS GUNADARMA 2025**

- **Nama      :** Ahmad Zaky Humami
- **NPM       :** 50422138

<p align="center">
    <img src="" width=500 alt="gunadarma_logo" />
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

Semenjak fase CRISP‑DM “Modeling”, arsitektur dan jumlah neuron pada tiap lapisan disesuaikan melalui eksperimen hyperparameter (jumlah unit, laju dropout, optimizer, dan batch size) untuk memaksimalkan akurasi prediksi. Model dilatih menggunakan algoritma Adam pada loss fungsi categorical_crossentropy, kemudian dievaluasi dengan metrik akurasi dan confusion matrix. Dengan pendekatan ini, jaringan saraf mampu mempelajari pola kompleks nilai rapor siswa dan memproyeksikannya ke rekomendasi tiga jurusan terbaik secara andal.


## Data Understanding
### Sumber Data
Dataset Nilai Akademik Siswa (__student_grades_5000.csv__)
Dataset ini disusun secara sintetis dengan bantuan model GPT untuk mensimulasikan kondisi nyata nilai rapor siswa SMA yang akan mendaftar ke Universitas Gunadarma. Nilai–nilai mata pelajaran pada dataset dibuat mengikuti skala 10–98, sesuai rentang penilaian Kurikulum Merdeka terbaru yang diterapkan oleh Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia. Setiap baris mewakili satu siswa, mencakup delapan mata pelajaran pokok (agama, PPKn, Bahasa Indonesia, Matematika, Bahasa Inggris, Seni Budaya, Penjaskes, Sejarah) serta mata pelajaran peminatan IPA atau IPS. Penciptaan data ini bertujuan memberikan distribusi nilai yang lebih variatif dan representatif, sekaligus mendekati pola sebaran akademik siswa dalam dunia nyata.

Dataset Jurusan Universitas Gunadarma (__ug_majors.csv__)
File ini memuat daftar 13 program studi di Universitas Gunadarma, lengkap dengan kode jurusan, nama fakultas, dan mata pelajaran terkait yang menjadi dasar rekomendasi. Selain itu, setiap jurusan dilengkapi kolom passing grade, yakni nilai ambang rata‑rata mata pelajaran terkait yang merefleksikan batas minimal kelayakan pendaftaran. Nilai passing grade diperoleh dari riset data historis dan kebijakan penerimaan di Universitas Gunadarma, lalu disisipkan secara eksplisit ke dalam dataset. Dengan demikian, dataset ini siap dipakai untuk memadukan rekomendasi model Neural Network dan penerapan aturan ambang kelulusan (threshold), sehingga menghasilkan sistem rekomendasi yang tidak hanya akurat secara prediksi, tapi juga realistis dalam konteks kebijakan seleksi perguruan tinggi.

### Deskripsi Variabel
#### Dataset Student Grades
| Variabel               | Keterangan                                                         |
| ---------------------- | ------------------------------------------------------------------ |
| `student_id`           | ID unik setiap siswa, format “S00001” hingga “S05000”              |
| `track`                | Jalur peminatan siswa: “IPA” atau “IPS”                            |
| **Core Subjects**      |                                                                    |
| `agama`                | Nilai mata pelajaran Pendidikan Agama                              |
| `ppkn`                 | Nilai mata pelajaran PPKn (Pendidikan Pancasila & Kewarganegaraan) |
| `bahasa_indonesia`     | Nilai mata pelajaran Bahasa Indonesia                              |
| `matematika`           | Nilai mata pelajaran Matematika                                    |
| `bahasa_inggris`       | Nilai mata pelajaran Bahasa Inggris                                |
| `seni_budaya`          | Nilai mata pelajaran Seni & Budaya                                 |
| `penjaskes`            | Nilai mata pelajaran Pendidikan Jasmani, Olahraga, dan Kesehatan   |
| `sejarah`              | Nilai mata pelajaran Sejarah Indonesia & Dunia                     |
| **Elective IPA**       | (kolom diisi jika `track` = “IPA”, else NaN)                       |
| `fisika`               | Nilai mata pelajaran Fisika                                        |
| `kimia`                | Nilai mata pelajaran Kimia                                         |
| `biologi`              | Nilai mata pelajaran Biologi                                       |
| `matematika_peminatan` | Nilai Matematika peminatan (khusus IPA)                            |
| `informatika`          | Nilai mata pelajaran Informatika                                   |
| **Elective IPS**       | (kolom diisi jika `track` = “IPS”, else NaN)                       |
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
...
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
Ekonomi  | 2127
Sosiologi  | 2127
Geografi  | 2127
Antropologi  | 2127
Sastra Indonesia  | 2127
Bahasa Asing  | 2127

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
Jumlah Duplikasi Data Rating : 0
```

### Exploratory Data Analysis (EDA)
##### Proporsi Jalur Peminatan (track)

<p align="center">
    <img src="" width=500 alt="track_distribution" />
</p>

Proporsi Jalur Peminatan (track):

##### Distribusi Nilai Mata Pelajaran Pokok

<p align="center">
    <img src="" width=500 alt="core_subjects" />
</p>

Distribusi Nilai Mata Pelajaran Pokok: 

##### Distribusi Nilai Mata Pelajaran IPA

<p align="center">
    <img src="" width=500 alt="ipa_subjects" />
</p>

Distribusi Nilai Mata Pelajaran IPA: 

##### Distribusi Nilai Mata Pelajaran IPS

<p align="center">
    <img src="" width=500 alt="ips_subjects" />
</p>

Distribusi Nilai Mata Pelajaran IPS:


## Data Preparation
### Feature Enginering
#### Recommended Major
Pertama saya melakukan Feature Engineering untuk kita menghitung skor potensi siswa untuk setiap jurusan berdasarkan nilai mata pelajaran terkait. Fungsi compute_major_score() akan menerima satu baris data siswa dan daftar mata pelajaran yang relevan untuk sebuah jurusan, lalu mengembalikan rata‑rata nilai dari mata pelajaran tersebut.

Setelah itu, untuk setiap siswa, kita melakukan iterasi ke seluruh daftar jurusan (df_major) dan hanya mempertimbangkan jurusan yang sesuai dengan jalur (IPA, IPS, atau IPA/IPS). Skor jurusan dihitung dengan memanggil compute_major_score(), kemudian jurusan dengan skor rata‑rata tertinggi dipilih sebagai rekomendasi rule‑based awal. Hasil akhir disimpan ke dalam kolom recommended_major pada df_student:
```
def compute_major_score(row, major_subjects):
      vals = [row[subj] for subj in major_subjects if pd.notna(row.get(subj))]
      return np.mean(vals) if vals else np.nan

recommended_codes = []
for _, student in df_student.iterrows():
      track = student['track'].lower()
      best_score = -np.inf
      best_code  = None

      for _, mj in df_major.iterrows():
            # Ambil track_type bisa 'IPA', 'IPS', atau 'IPA/IPS'
            mj_track = mj['track_type'].lower()
            if (mj_track == track) or (mj_track == 'ipa/ips'):
                  # pecah related_subjects dengan ';'
                  subjects = [s.strip() for s in mj['related_subjects'].split(';')]
                  score = compute_major_score(student, subjects)
                  if score > best_score:
                        best_score = score
                        best_code  = mj['code']

      recommended_codes.append(best_code)

df_student['recommended_major'] = recommended_codes
```
Output Distribusi Recommended Major :
```
recommended_major
A1     225
A10    565
A11    507
A12    540
A13    470
A2     479
A3      52
A4     719
A5     290
A6     281
A7     386
A9     388
Name: count, dtype: int64
```

#### Threshold Passing Grade
Lanjut saya menambahkan langkah perhitungan passing grade pada fase Data Preparation untuk memastikan rekomendasi jurusan tidak sekadar berdasar peringkat skor saja, tetapi juga merefleksikan ambang kompetisi yang riil. Dengan fungsi `add_passing_grade()`, saya menghitung persentil ke‑75 dari distribusi rata‑rata nilai siswa pada mata pelajaran terkait untuk setiap jurusan. Persentil ini saya ambil sebagai threshold—artinya, saya menganggap seorang calon siswa “memenuhi syarat” jika rata‑rata nilai pada mata pelajaran jurusan tersebut berada di atas nilai yang dicapai 75% siswa dalam dataset sintetis.

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
            # Parse dan format nama mapel terkait
            rel = [s.strip().lower().replace(' ', '_')
                  for s in mj[related_col].split(';')]
            # Ambil kolom yang cocok
            cols = [c for c in df_students.columns if c in rel]
            if not cols:
                  thresholds.append(np.nan)
                  continue
            # Rata‑rata per siswa
            avg_scores = df_students[cols].mean(axis=1)
            # Persentil ke-pct
            value = np.percentile(avg_scores.dropna(), pct)
            thresholds.append(round(value, 2))
      return pd.Series(thresholds, name='passing_grade')

df_major['passing_grade'] = add_passing_grade(df_student, df_major, pct=75)
```
Output :

<p align="center">
    <img src="" width=500 alt="passing_grades" />
</p>


### Mapping Track
Lanjut saya menambahkan kolom `track` di `df_student` untuk mencerminkan peminatan IPA atau IPS siswa. Pada fase Data Preparation, saya melakukan one‑hot encoding sederhana dengan mengubah nilai string "IPA" menjadi 1 dan "IPS" menjadi 0, lalu menyimpannya di kolom track_bin. Langkah ini mempermudah model Neural Network membedakan jalur peminatan tanpa perlu memproses teks secara langsung. Dengan demikian, setiap baris data siswa kini memiliki fitur numerik track_bin yang digunakan bersama nilai mata pelajaran untuk melatih dan melakukan inferensi pada model.
```
df_student['track_bin'] = df_student['track'].map({'IPA': 1, 'IPS': 0}).astype(int)
final_df = df_student.drop(columns=['student_id','track'])
```
Ouput : 

<p align="center">
    <img src="" width=500 alt="mapping_track" />
</p>


### Label Encoder
Pada tahap Label Encoding, saya mengonversi target rekomendasi jurusan (__recommended_major__) yang awalnya berupa kode string seperti “A1”, “A2”, …, “A13” menjadi format numerik sehingga dapat ditangani oleh model. Saya menggunakan LabelEncoder dari scikit‑learn untuk memetakan setiap kode jurusan ke indeks bilangan bulat unik. Setelah proses encoding, variabel target (y) berwujud array integer, sementara daftar classes menyimpan urutan kode jurusan asli. Langkah ini memastikan kompatibilitas dengan fungsi predict() dan metrik evaluasi pada model Neural Network Keras Sequential, sekaligus mempertahankan kemampuan untuk melakukan inverse transform saat menampilkan kembali kode jurusan hasil prediksi.
```
le = LabelEncoder().fit(final_df['recommended_major'])
classes = list(le.classes_)
```

### Numerical Features & Split Data
Saya memilih 19 fitur numerik—terdiri dari delapan mata pelajaran pokok dan sebelas mata pelajaran peminatan—sebagai variabel input utama. Setelah mengumpulkan data siswa dan target rekomendasi jurusan, saya melakukan train–validation–test split secara stratified (70% train, 15% validation, 15% test) berdasarkan label jurusan. Pendekatan stratified ini menjaga distribusi setiap jurusan tetap konsisten di ketiga subset, sehingga evaluasi akurasi dan metrik lain mencerminkan performa model pada populasi yang representatif.
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
Untuk menangani missing values—karena perbedaan jalur IPA/IPS yang membuat beberapa kolom “kosong”—saya menggunakan SimpleImputer(strategy='mean') untuk mengganti nilai NaN dengan rata‑rata kolom, diikuti dengan StandardScaler() untuk melakukan normalisasi z‑score (mean = 0, std = 1). Kedua langkah ini saya bungkus dalam Pipeline sehingga preprocessing dapat diterapkan konsisten pada data train, validation, dan test, serta pada data inference baru.
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
      patience=10,
      restore_best_weights=True
)

history = model.fit(
      X_train_scaled, y_train,
      validation_split=0.15,
      validation_data=(X_val_scaled, y_val),
      epochs=100,
      batch_size=32,
      callbacks=[callbacks],
      verbose=2
)
```

<p align="center">
    <img src="" width=500 alt="neuralnetwork_sequential" />
</p>

Model ini saya kompilasi dengan optimizer Adam, loss categorical_crossentropy, dan metrik accuracy. Selain itu, saya menambahkan callback EarlyStopping (monitor=‘val_loss’, patience=10) agar pelatihan berhenti otomatis saat tidak ada peningkatan, meminimalkan overfitting dan menghemat waktu komputasi.

### SHAP
SHAP (SHapley Additive exPlanations) Summary Plot menampilkan kontribusi rata‑rata absolut setiap fitur terhadap prediksi model, diukur sebagai nilai SHAP. Sumbu horizontal menggambarkan “mean(|SHAP value|)”: semakin panjang batang, semakin besar peran fitur tersebut dalam memengaruhi keputusan model. Fitur–fitur diurutkan dari yang paling penting (puncak grafik) hingga yang paling tidak berpengaruh (bawah grafik).

<p align="center">
    <img src="" width=500 alt="shap_graph" />
</p>

Warna pada setiap batang mewakili nilai aktual fitur: titik‑titik merah menandakan sampel dengan nilai fitur tinggi, biru menandakan sampel dengan nilai rendah. Misalnya, pada grafik di atas fitur informatika dan seni_budaya berada di urutan teratas, menunjukkan bahwa variasi nilai kedua mata pelajaran ini paling banyak menggeser probabilitas rekomendasi jurusan; siswa dengan nilai informatika tinggi lebih cenderung dipetakan ke jurusan IT, sedangkan nilai seni_budaya tinggi mendorong rekomendasi jurusan–jurusan berbasis estetika (Desain Interior, Arsitektur). Dengan demikian, SHAP memberikan gambaran kuantitatif sekaligus intuitif tentang mana mata pelajaran yang paling krusial bagi model dalam menentukan jurusan terbaik bagi setiap siswa.


### LIME
LIME (Local Interpretable Model‑agnostic Explanations) LIME membantu kita memahami “mengapa” model Neural Network mengambil keputusan tertentu untuk tiap sampel siswa. Daripada melihat keseluruhan dataset, LIME membuat model lokal sederhana (biasanya regresi linier) di sekitar titik data yang ingin dijelaskan. Pada grafik LIME:

<p align="center">
    <img src="" width=500 alt="lime_graph" />
</p>

1. Panel kiri memperlihatkan probabilitas prediksi masing‑masing jurusan, sehingga Anda tahu seberapa yakin model.

2. Diagram batang di tengah (bar chart) menampilkan lima fitur (mata pelajaran) teratas yang paling banyak mendorong prediksi ke satu jurusan—misalnya A10 (Psikologi). Batang ke kanan berarti fitur tersebut “menambah” probabilitas jurusan; batang ke kiri berarti “mengurangi.”

3. Tabel di kanan memuat nilai Z‑score (baku) fitur pada sampel: semakin positif nilainya (mis. bahasa_indonesia 1.69), semakin besar dorongan ke prediksi, dan sebaliknya.

Dengan LIME, kita mendapatkan penjelasan yang lokal dan intuitif: untuk setiap siswa, kita dapat melihat mata pelajaran mana yang menjadi “bintang” pendorong rekomendasi model, serta mana yang justru menahan probabilitasnya. Penjelasan ini memudahkan siswa maupun orang tua memahami dan percaya pada rekomendasi jurusan yang dihasilkan.

## Tuning Model
### RandomizedSearchCV
Saya menerapkan __RandomizedSearchCV__ untuk menyetel hyperparameter Neural Network Keras saya secara efisien dan terukur. Alih‑alih mencoba setiap kombinasi parameter secara exhaustif yang bisa memakan waktu puluhan jam tetapi dengan __RandomizedSearchCV__ secara acak mengambil ratusan konfigurasi dari ruang pencarian yang telah saya tentukan (misalnya rentang learning rate, dropout rate, jumlah unit lapisan), lalu mengevaluasi masing‑masing melalui cross‑validation. Dengan cara ini, model diuji pada beberapa “lipatan” data yang berbeda, sehingga saya mendapatkan gambaran stabil tentang performanya. Setelah seluruh iterasi selesai, __RandomizedSearchCV__ memilih kombinasi dengan skor validasi tertinggi sebagai hyperparameter final. Pendekatan acak ini tidak hanya jauh lebih cepat daripada grid search konvensional, tetapi juga mampu menjelajahi area hyperparameter yang mungkin luput dari grid search yang kaku.

#### Define Parameter
```
keras_clf = KerasClassifier(
      model       = build_model,
      validation_split = 0.15,
      epochs      = 100,
      batch_size  = 32,
      callbacks   = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
      verbose     = True,
)

param_dist = {
      "model__learning_rate": loguniform(1e-5, 1e-1),  
      "model__first_units":    randint(32, 256),
      "model__second_units":   randint(16, 128),
      "model__optimizer":      ["adam", "rmsprop"],
      "model__activation":     ["relu", "elu", "gelu", "tanh"],
      "model__dropout_rate":   [0.1, 0.2, 0.3, 0.4],
      "batch_size":            [16, 32, 64]
}
```
#### Train Hyperparameter
```
for stage in range(4):  # 4 × 50 = 200 iterasi total
      random_search = RandomizedSearchCV(
            estimator = keras_clf,
            param_distributions = param_dist,
            n_iter = 50,
            cv = 3,
            scoring = "accuracy",
            random_state = 42 + stage,
            return_train_score = True,  # Tambahkan ini untuk mendapatkan skor pelatihan
            verbose = True,  # Tambahkan verbosity untuk melihat progres
            n_jobs = -1,  # Gunakan semua core CPU
      )

      random_search.fit(X_train_scaled, y_train)

      if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_params_overall = random_search.best_params_

```

<p align="center">
    <img src="" width=500 alt="shapcv_graph" />
</p>

<p align="center">
    <img src="" width=500 alt="limecv_graph" />
</p>

Insight dari hasil tuning:


## Evaluasi
### Evaluasi Model Keras Sequential (Sebelum Tuning)

<p align="center">
    <img src="" width=500 alt="cm_model" />
</p>
Confusion matrix awal menunjukkan model mampu mengenali sebagian besar kelas jurusan dengan baik, terutama untuk kelas A4 (Teknik Informatika) yang terdeteksi 105 sampel dengan akurasi tinggi. Namun ada beberapa titik kebingungan (“off‑diagonal”) yang konsisten, misalnya 3 sampel A1 (Manajemen) terprediksi sebagai A13 (Desain Interior) dan 2 sampel A11 (Arsitektur) terdeteksi sebagai A4. Kelas A3 (Teknik Industri) relatif sedikit sampelnya sehingga terjadi mis‑classifikasi ke A7 (Sistem Informasi) meski hanya 2 kasus. Keseluruhan, rata‑rata akurasi tiap kelas berkisar antara 75–98%, menandakan model sudah belajar pola fitur akademik, tetapi masih butuh penyempurnaan pada jurusan dengan jumlah data terbatas atau feature overlap tinggi.

### Evaluasi Model Keras Sequential Terbaik (Setelah Tuning)

<p align="center">
    <img src="" width=500 alt="cm_best_model" />
</p>
Confusion matrix model terbaik memperlihatkan peningkatan keseragaman diagonal: misalnya kelas “Code 8” (Ilmu Komputer) kini terdeteksi 107/109 sampel dengan benar, naik dari 105 sebelumnya. Kesalahan prediksi untuk kelas “Code 1” (Manajemen) yang awalnya 3 sample menurun menjadi 2, dan confusion antara A1/A2 berkurang secara signifikan. Beberapa kelas seperti “Code 3” (Teknik Industri) dan “Code 10” (Psikologi) juga menunjukkan peningkatan ketepatan prediksi. Secara keseluruhan, tuning hyperparameter melalui RandomizedSearchCV berhasil meningkatkan stabilitas dan konsistensi model, mengurangi noise di cell‑cell off‑diagonal, serta memperkuat kapabilitas model dalam membedakan jurusan yang fitur‑fiturnya saling tumpang‑tindih.

### Interpretasi Hasil dan Implikasi
Perbandingan kedua confusion matrix menunjukkan bahwa teknik tuning secara acak (RandomizedSearchCV) tidak hanya meningkatkan skor validasi, tetapi juga konkret mengurangi kesalahan klasifikasi spesifik—terutama di jurusan teknik dan teknologi informasi. Hal ini menegaskan keefektifan pendekatan Keras Sequential untuk memetakan profil nilai akademik siswa ke jurusan target di Universitas Gunadarma. Adanya residual error di beberapa jurusan menandakan perlunya perbaikan data (misalnya menambah sampel di jurusan under‑represented) atau eksplorasi fitur tambahan (seperti data non‑akademik) untuk menyempurnakan sistem rekomendasi.

### Kesimpulan Evaluasi
Model terbaik mencapai keseimbangan optimal antara kompleksitas dan generalisasi: confusion matrix yang lebih “bersih” memvalidasi bahwa pemilihan arsitektur layer, fungsi aktivasi, learning rate, dan dropout rate berhasil memfokuskan model pada pola distribusi nilai yang relevan. Dengan akurasi per‑kelas yang meningkat dan false positive yang menurun, sistem ini kini siap diintegrasikan pada antarmuka Streamlit untuk mendukung siswa dalam memilih jurusan kuliah—memberikan tidak hanya prediksi probabilistik, tetapi juga keyakinan empiris yang kuat berdasarkan hasil evaluasi.

## Reference
- Badan Pusat Statistik. 2022. Statistik Pendidikan 2022. Badan Pusat Statistik. Jakarta.

- Muttaqin et al. 2023. Pengenalan Data Mining. Yayasan Kita Menulis, Yogyakarta.

- Purwati, Neni, Rini Nurlistiani dan Oscar Devinsen. 2020. ‘Data Mining dengan Algoritma Neural Network dan Visualisasi Data untuk Prediksi Kelulusan Mahasiswa’. Jurnal Informatika, 20(2), hlm. 156–163.

- Raihan, Muhammad Iqbal. 2024. ‘Implementasi Backpropagation untuk Rekomendasi Jurusan Peminatan Mahasiswa Program Studi Teknik Informatika di Universitas Islam Balitar’. Jurnal Ilmiah Sistem Informasi dan Teknik Informatika (JISTI), 7(2), hlm. 219–226.

- Rianti, Afika, Nuur Wachid Abdul Majid dan Ahmad Fauzi. 2023. ‘CRISP-DM: Metodologi Proyek Data Science’. Prosiding Seminar Nasional Teknologi Informasi dan Bisnis (SENATIB) 2023.

- Siardizal dan Rosy Dasmita. 2016. ‘Jaringan Syaraf Tiruan untuk Menentukan Jurusan di Sekolah Menengah Atas (SMA) (Studi Kasus: SMA Negeri 1 Sungai Penuh)’. Jurnal Ilmu Pengetahuan dan Sistem Informasi, 2: 91–100.

- Ulfa, Annisa, Doni Winarso dan Edo Arribe. 2020. ‘Sistem Rekomendasi Jurusan Kuliah Bagi Calon Mahasiswa Baru Menggunakan Algoritma C4.5’. Jurnal FASILKOM, 10(1): 61–65.

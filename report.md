# Laporan Eksperimen Recurrent Neural Network (RNN) untuk Klasifikasi Sentimen dan Text Summarization
Nama : Muhammad Rizkiana Pratama

NIM : 2404421

## 1. Pendahuluan

### 1.1 Latar Belakang

Analisis sentimen merupakan salah satu tugas *natural language processing* (NLP) yang bertujuan menentukan kecenderungan opini dalam suatu teks, misalnya positif, negatif, atau netral. Pada media sosial, opini terhadap film ditulis secara singkat, informal, dan sering mengandung singkatan, mention, tagar, emoji, serta tanda baca yang tidak baku. Kondisi tersebut membuat proses preprocessing dan pemilihan model menjadi bagian penting dalam membangun sistem klasifikasi sentimen.

Text summarization adalah tugas menghasilkan ringkasan pendek dari sebuah dokumen sumber dengan tetap mempertahankan informasi utama. Salah satu pendekatan klasik untuk tugas ini adalah **Sequence-to-Sequence (Seq2Seq)**, yaitu model encoder-decoder yang memetakan sequence input (artikel) menjadi sequence output (ringkasan).

*Recurrent Neural Network* (RNN) merupakan arsitektur jaringan saraf yang dirancang untuk memproses data berurutan. Pada setiap langkah, RNN membaca satu token dan memperbarui *hidden state* yang menyimpan informasi dari token-token sebelumnya. Sifat ini membuat RNN sesuai untuk teks karena makna sebuah kata dapat bergantung pada konteks kata-kata di sekitarnya.

Kedua eksperimen dalam laporan ini menggunakan dua bentuk pemetaan RNN yang berbeda:

- **Many-to-one** pada klasifikasi sentimen: seluruh urutan token dalam satu tweet diproses menjadi satu *hidden state* akhir, kemudian dipetakan menjadi satu label sentimen. Dengan demikian, satu tweet menghasilkan tepat satu prediksi: `negative` atau `positive`.
- **Many-to-many** pada summarization: encoder memproses artikel, lalu decoder menghasilkan satu token ringkasan pada setiap langkah, sehingga output berupa sequence.

Pada eksperimen summarization digunakan sel **Vanilla RNN** murni. Setiap langkah decoder menghasilkan satu token, sehingga output berupa sequence ringkasan. Berbeda dengan LSTM/GRU, Vanilla RNN hanya memiliki satu hidden state sederhana:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

Keterbatasan utama Vanilla RNN adalah vanishing gradient dan bottleneck pada satu vektor hidden state. Eksperimen summarization bertujuan melihat perilaku tersebut secara empiris melalui perbandingan beberapa konfigurasi optimizer, learning rate, dan hidden dimension. Eksperimen klasifikasi melengkapi gambaran tersebut dengan membandingkan SimpleRNN terhadap varian bergating (GRU/LSTM) dan varian bidirectional pada dataset yang kecil.

### 1.2 Tujuan

Laporan ini mencakup dua kelompok eksperimen dengan tujuan berikut.

**A. Klasifikasi sentimen tweet opini film**

1. Membaca dan memeriksa dataset sentimen tweet opini film dari file lokal.
2. Melakukan preprocessing teks dasar, meliputi *cleaning*, *case folding*, dan penghapusan karakter khusus.
3. Membagi dataset menjadi data latih, data validasi, dan data uji dengan proporsi yang terkontrol.
4. Membangun model RNN many-to-one menggunakan TensorFlow/Keras.
5. Melatih model dan menampilkan kurva *loss* serta *accuracy* selama proses training.
6. Mengevaluasi model pada data uji menggunakan *confusion matrix* dan *classification report* yang mencakup accuracy, precision, recall, dan F1-score.
7. Membandingkan optimizer, arsitektur RNN, dan beberapa hyperparameter melalui eksperimen tuning dengan beberapa random seed.
8. Menganalisis performa model, keterbatasan, dan peluang pengembangannya.

**B. Text summarization**

1. Mengimplementasikan pipeline lengkap text summarization berbasis Seq2Seq Vanilla RNN dengan TensorFlow/Keras.
2. Melakukan preprocessing teks Bahasa Indonesia, pembentukan vocabulary, encoding, dan padding.
3. Membandingkan minimal tiga konfigurasi eksperimen (optimizer, learning rate, dan hidden dimension).
4. Mencatat training loss, validation loss, dan execution time per epoch untuk setiap konfigurasi.
5. Melakukan inference autoregressive dengan fungsi `summarize(text)`.
6. Menghitung ROUGE-1, ROUGE-2, dan ROUGE-L pada data uji agar hasilnya siap dimasukkan ke tabel laporan.

### 1.3 Ruang Lingkup Eksperimen

| Aspek | Klasifikasi Sentimen | Text Summarization |
|---|---|---|
| Dataset | `dataset_tweet_sentiment_opini_film.csv` | `wikihow.json` |
| Bentuk tugas | Many-to-one | Many-to-many (Seq2Seq) |
| Input → output | Tweet → label sentimen | Artikel → ringkasan |
| Model utama | SimpleRNN 32 unit | Seq2Seq SimpleRNN, hidden 128/256 |
| Model pembanding | GRU, LSTM, BiGRU, BiLSTM, BiSimpleRNN | 4 konfigurasi optimizer/learning rate/hidden |
| Jumlah data | 197 tweet (133 latih / 24 validasi / 40 uji) | 6.011 pasangan (4.793 / 600 / 618) |
| Metrik utama | Accuracy, precision, recall, F1, confusion matrix | ROUGE-1, ROUGE-2, ROUGE-L, loss |
| Durasi training | Maksimum 60 epoch dengan early stopping | 8 epoch |
| Artefak | `klasifikasi/output_rnn/` | `summarize/outputs/` |

---

## 2. Dataset

### 2.1 Dataset Klasifikasi Sentimen Tweet Opini Film

Dataset yang digunakan adalah `klasifikasi/dataset_tweet_sentiment_opini_film.csv`. Dataset berisi opini berbahasa Indonesia mengenai film dan memiliki tiga kolom:

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Id` | Integer | Identitas unik setiap tweet |
| `Sentiment` | String | Label sentimen: `negative` atau `positive` |
| `Text Tweet` | String | Isi tweet opini film |

#### Karakteristik data

| Statistik | Nilai |
|---|---:|
| Jumlah data mentah | 200 tweet |
| Jumlah kelas | 2 (biner) |
| Distribusi awal | 100 negative, 100 positive |
| Jumlah data setelah pembersihan duplikat | 197 tweet |
| Distribusi final | 99 negative, 98 positive |
| Panjang tweet | 2–26 kata |
| Rata-rata panjang tweet | 15,11 kata |
| Median panjang tweet | 16 kata |
| Rata-rata jumlah karakter | 97,9 karakter |

#### Contoh data

| Id | Sentiment | Text Tweet |
|---:|---|---|
| 1 | negative | Jelek filmnya... apalagi si ernest gak mutu bgt actingnya... film sampah |
| 102 | positive | Habis nonton film Kong: Skull Island 2017. Lumayan greget filmnya, 7.5 deh.. layak buat nobar |
| 103 | positive | Bravo untuk #filmkartini! Pemainnya bagus, sudut pandang ceritanya pun menarik. Film yang penting untuk ditonton perempuan dan juga lelaki. |

Dataset bersifat seimbang sejak awal. Setelah teks dibersihkan, ditemukan tiga data duplikat sehingga jumlah data menjadi 197 dengan komposisi 99 tweet negatif dan 98 tweet positif. Keseimbangan ini membuat metrik accuracy dapat digunakan sebagai gambaran awal performa tanpa bias kelas yang ekstrem, meskipun precision, recall, dan F1-score tetap perlu dilaporkan.

Tantangan utama dataset adalah sifat teks media sosial yang tidak baku, misalnya penggunaan kata seperti `gak`, `bgt`, `yg`, `abis`, mention akun, tagar, dan emoji. Selain itu, jumlah data relatif kecil untuk melatih model *deep learning* dari awal.

### 2.2 Dataset Text Summarization WikiHow

Dataset yang digunakan adalah `summarize/wikihow.json` dengan struktur:

```json
[
  {
    "pageName": "Agar-Akur-dengan-Pasangan",
    "steps": [
      {
        "abstract": "Fokuslah pada hal positif.",
        "article": "Penting untuk meluangkan waktu untuk menghargai pasangan ..."
      }
    ]
  }
]
```

Setiap `step` diposisikan sebagai satu pasangan data:

- **Input encoder:** `article` (penjelasan/langkah)
- **Target decoder:** `abstract` (ringkasan singkat)

Statistik dataset penuh:

| Item | Nilai |
|---|---:|
| Jumlah halaman | 25.630 |
| Jumlah pasangan `article`-`abstract` | 365.121 |
| Rata-rata token artikel | 36,8 |
| Median token artikel | 34 |
| Persentil ke-90 token artikel | 62 |
| Rata-rata token abstract | 5,4 |
| Median token abstract | 5 |
| Persentil ke-90 token abstract | 10 |

Karena keterbatasan komputasi (CPU-only), eksperimen utama menggunakan **6.011 pasangan** dari 512 halaman yang dipilih secara acak dengan seed 42. Kode mendukung seluruh dataset dengan mengubah `--max-samples 0`.

**Pembagian data (split per halaman agar tidak bocor):**

| Split | Jumlah pasangan |
|---|---:|
| Train | 4.793 |
| Validation | 600 |
| Test | 618 |
| **Total** | **6.011** |

Pembagian per halaman memastikan langkah-langkah dari halaman yang sama tidak muncul di train dan test sekaligus.

---

## 3. Eksperimen

### 3.1 Klasifikasi Sentimen

#### 3.1.1 Alur Eksperimen

Alur eksperimen klasifikasi secara ringkas:

```text
Dataset CSV
    ↓
Preprocessing teks (cleaning, case folding, hapus karakter khusus)
    ↓
Pembersihan data kosong dan duplikat
    ↓
Train-test split (stratified)
    ↓
Vektorisasi teks (TextVectorization, fit pada data latih)
    ↓
Embedding → SimpleRNN → Dense → Sigmoid
    ↓
Training + validasi
    ↓
Evaluasi pada data uji
    ↓
Kurva training, confusion matrix, classification report
```

Lingkungan eksperimen dirangkum pada Lampiran C. Seluruh eksperimen dijalankan di CPU. Eksperimen utama menggunakan random seed 42, sedangkan eksperimen tuning klasifikasi menggunakan seed 42, 123, dan 2024.

#### 3.1.2 Preprocessing Teks

Preprocessing dilakukan untuk mengurangi noise tanpa menghilangkan informasi sentimen utama. Tahapan yang diterapkan adalah:

1. **Cleaning**
   - Menghapus URL dan tautan `www`.
   - Menghapus mention seperti `@username`.
   - Menghapus penanda retweet `RT`.
   - Mengubah tagar `#kata` menjadi `kata` agar isi tagar tetap dipertahankan.
   - Mengubah HTML entity seperti `&amp;` menjadi karakter aslinya.
2. **Case folding**
   - Mengubah seluruh teks menjadi huruf kecil.
3. **Penghapusan karakter khusus**
   - Menghapus angka, tanda baca, emoji, simbol, dan karakter selain huruf `a–z` serta spasi.
4. **Normalisasi spasi**
   - Mengganti spasi berlebih menjadi satu spasi dan memangkas spasi di awal/akhir teks.
5. **Pembersihan data**
   - Membuang tweet yang menjadi kosong setelah preprocessing.
   - Membuang duplikat berdasarkan teks yang sudah dibersihkan.

Contoh hasil preprocessing:

```text
Sebelum : @beexkuanlin Sepanjang film gwa berkata kasar terus pada bapaknya
Sesudah : sepanjang film gwa berkata kasar terus pada bapaknya

Sebelum : Banyak yg kecewa abis nonton film ini :(
Sesudah : banyak yg kecewa abis nonton film ini
```

Preprocessing ini sengaja dibatasi pada tahap dasar. Stemming, penghapusan stopword, normalisasi kata gaul, dan penanganan negasi tidak diterapkan agar eksperimen tetap sesuai dengan ketentuan preprocessing dasar. Konsekuensinya, variasi kata seperti `gak` dan `tidak` tetap dianggap sebagai token yang berbeda.

#### 3.1.3 Pembagian Data

Dataset dibagi menggunakan `train_test_split` dari scikit-learn dengan `random_state=42` dan stratifikasi berdasarkan label. Stratifikasi memastikan proporsi kelas negatif dan positif tetap seimbang pada setiap subset.

| Subset | Negative | Positive | Total | Proporsi dari 197 |
|---|---:|---:|---:|---:|
| Data latih | 67 | 66 | 133 | 67,5% |
| Data validasi | 12 | 12 | 24 | 12,2% |
| Data uji | 20 | 20 | 40 | 20,3% |
| **Total** | **99** | **98** | **197** | **100%** |

Data validasi diambil dari sebagian data latih, bukan dari data uji. Proporsi data uji adalah 20% dari total data, sedangkan data validasi adalah 15% dari 157 data latih (sebelum pemisahan validasi), sehingga diperoleh 24 tweet validasi. Dengan demikian, data uji tetap murni dan hanya digunakan satu kali pada evaluasi akhir. Data validasi dimanfaatkan untuk memantau performa selama training dan menentukan kapan proses training dihentikan.

#### 3.1.4 Vektorisasi Teks

Teks yang sudah dibersihkan diubah menjadi urutan bilangan bulat menggunakan layer `TextVectorization` dari Keras dengan konfigurasi:

| Parameter | Nilai |
|---|---:|
| Strategi pemisahan token | Whitespace |
| Panjang sequence | 23 token |
| Ukuran kosakata hasil adapt | 845 token |
| Batas maksimum kosakata | 10.000 |
| Padding | 0 |
| Token di luar kosakata | `[UNK]` |

Panjang sequence 23 ditentukan dari persentil ke-95 jumlah kata pada data latih. Nilai tersebut kemudian dibatasi pada rentang 10–60 token. Layer vektorisasi hanya di-*adapt* pada data latih untuk mencegah *data leakage* dari data validasi maupun data uji.

Bentuk data setelah vektorisasi:

- Data latih: `(133, 23)`
- Data validasi: `(24, 23)`
- Data uji: `(40, 23)`

#### 3.1.5 Arsitektur Model

Model yang digunakan adalah RNN many-to-one dengan detail sebagai berikut:

| Layer | Output Shape | Jumlah Parameter | Keterangan |
|---|---:|---:|---|
| Input `token_ids` | `(None, 23)` | 0 | Sequence token integer |
| `Embedding` | `(None, 23, 64)` | 54.080 | Dimensi embedding 64, regularisasi L2 |
| `SimpleRNN` | `(None, 32)` | 3.104 | 32 unit, `return_sequences=False` |
| `Dense` | `(None, 32)` | 1.056 | Aktivasi ReLU |
| `Dropout` | `(None, 32)` | 0 | Dropout 0.5 |
| `Dense` output | `(None, 1)` | 33 | Aktivasi Sigmoid |
| **Total parameter trainable** |  | **58.273** |  |

Karena layer SimpleRNN menggunakan `return_sequences=False`, model hanya mengambil *hidden state* terakhir dari seluruh urutan. Inilah yang membuat arsitektur tersebut termasuk **many-to-one**: banyak token masukan menghasilkan satu keluaran.

Beberapa teknik regularisasi digunakan karena dataset hanya berisi 197 tweet:

- Dropout sebesar 0.5 pada input RNN.
- Recurrent dropout sebesar 0.2 pada koneksi rekuren.
- Dropout sebesar 0.5 pada layer dense.
- Regularisasi L2 sebesar `1e-4` pada matriks embedding.

Model dikompilasi dengan:

| Komponen | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate awal | 0.0005 |
| Fungsi loss | Binary crossentropy |
| Metrik | Accuracy |
| Label | `negative = 0`, `positive = 1` |
| Ambang prediksi | 0.5 |

#### 3.1.6 Konfigurasi Pelatihan

| Parameter | Nilai |
|---|---:|
| Random seed | 42 |
| Maksimum epoch | 60 |
| Batch size | 16 |
| Early stopping | Monitor `val_loss`, patience 10, `min_delta=1e-4` |
| Restore best weights | Ya |
| ReduceLROnPlateau | Factor 0.5, patience 4, minimum learning rate `1e-5` |
| Model checkpoint | Monitor `val_loss`, simpan model terbaik |

Selama training, performa data latih dan data validasi dicatat. Setelah training selesai, bobot model dikembalikan ke epoch dengan `val_loss` terbaik. Model terbaik juga disimpan sebagai `klasifikasi/output_rnn/model_rnn_terbaik.keras`.

Training berhenti pada epoch ke-36 setelah `val_loss` tidak lagi membaik sesuai patience 10. Nilai `val_loss` terbaik adalah **0,5132** pada epoch ke-27.

#### 3.1.7 Metrik Evaluasi

Evaluasi dilakukan pada 40 data uji yang tidak pernah digunakan selama training maupun pemilihan model. Probabilitas keluaran sigmoid diubah menjadi kelas dengan ambang 0.5:

```text
probabilitas >= 0.5  → positive
probabilitas < 0.5   → negative
```

Metrik yang digunakan:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

Karena data uji seimbang, nilai *weighted average* dan *macro average* untuk precision, recall, dan F1-score identik. Laporan tetap menampilkan metrik per kelas agar perilaku model terhadap masing-masing sentimen dapat dianalisis.

#### 3.1.8 Eksperimen Pembanding dan Tuning Hyperparameter

Untuk memperkuat analisis dan membandingkan beberapa pilihan desain model, dilakukan eksperimen tambahan yang diimplementasikan pada `klasifikasi/eksperimen_tuning.py`. Eksperimen ini menggunakan split data, vektorisasi, dan protokol evaluasi yang sama dengan eksperimen utama agar hasilnya dapat dibandingkan secara adil.

Strategi eksperimen:

- Data latih, validasi, dan uji tetap 133, 24, dan 40 tweet.
- Setiap konfigurasi dijalankan dengan **tiga random seed**, yaitu 42, 123, dan 2024, sehingga diperoleh rata-rata dan simpangan baku performa validasi.
- Seluruh konfigurasi menggunakan early stopping, ReduceLROnPlateau, dan hyperparameter baseline yang sama; hanya faktor yang sedang diuji yang diubah.
- Pemilihan konfigurasi terbaik dilakukan berdasarkan **rata-rata validation F1-score** dari tiga seed, dengan validation accuracy dan validation loss sebagai *tie-breaker*.
- Data uji tidak digunakan untuk memilih konfigurasi. Data uji hanya dipakai sekali untuk mengevaluasi konfigurasi terbaik pada akhir eksperimen.
- Terdapat **23 konfigurasi unik** dan **69 run training** (23 konfigurasi × 3 seed).

Faktor yang diuji:

| Kelompok | Nilai yang diuji | Nilai baseline |
|---|---|---|
| Optimizer | Adam, AdamW, RMSprop, SGD | Adam, learning rate 5e-4 |
| Arsitektur RNN | SimpleRNN, GRU, LSTM, BiSimpleRNN, BiGRU, BiLSTM | SimpleRNN 32 unit |
| Jumlah unit RNN | 16, 32, 64 | 32 |
| Dimensi embedding | 32, 64, 128 | 64 |
| Dropout RNN | 0.3, 0.5, 0.7 | 0.5 |
| Recurrent dropout | 0.0, 0.2, 0.4 | 0.2 |
| Batch size | 8, 16, 32 | 16 |
| Learning rate | 1e-4, 5e-4, 1e-3 | 5e-4 |
| Regularisasi L2 embedding | 0, 1e-4, 1e-3 | 1e-4 |

Pada perbandingan optimizer, learning rate disesuaikan dengan karakter masing-masing optimizer agar perbandingan tidak bias: Adam dan AdamW menggunakan 5e-4, RMSprop 1e-3, serta SGD 5e-3 dengan momentum 0.9 dan Nesterov. AdamW menggunakan weight decay 1e-4. Pengaturan ini dipertahankan sama untuk seluruh seed pada optimizer yang bersangkutan.

Seluruh hasil per run, ringkasan statistik, dan grafik perbandingan disimpan di folder `klasifikasi/output_rnn/tuning/`. Hasil yang sudah dipecah per kelompok eksperimen (karena baseline muncul di beberapa kelompok) disimpan di `klasifikasi/output_rnn/tuning/hasil_tuning_per_grup.csv`.

### 3.2 Text Summarization

#### 3.2.1 Preprocessing dan Pembentukan Data

Konfigurasi preprocessing:

- Pembersihan HTML entity/tag, URL, email, karakter non-umum, dan whitespace.
- Lowercase.
- Tokenisasi sederhana Bahasa Indonesia menggunakan regex kata Unicode `\w+`.
- Tidak dilakukan stemming/stopword removal agar informasi tetap utuh.
- Vocabulary dibangun **hanya dari train set**.
- Token khusus: `<PAD>` (0), `<UNK>` (1), `<SOS>` (2), `<EOS>` (3).
- Ukuran vocabulary aktual: **11.781 token**.
- Panjang maksimum artikel: **128 token**.
- Panjang maksimum sequence decoder: **26 token** (termasuk `<SOS>` dan `<EOS>`).
- Padding menggunakan `<PAD>`.
- Split dilakukan berdasarkan halaman, sehingga langkah-langkah dari halaman yang sama tidak muncul di train dan test sekaligus.

#### 3.2.2 Arsitektur Model

Implementasi terdapat pada `summarize/model.py`.

**Encoder**

```text
Input artikel (integer sequence)
        ↓
Embedding(vocab_size, embedding_dim, mask_zero=True)
        ↓
SimpleRNN(hidden_dim, activation="tanh", return_state=True)
        ↓
hidden state terakhir h_T
```

`mask_zero=True` penting agar RNN mengabaikan posisi `<PAD>`; tanpa masking, hidden state terakhir didominasi oleh padding dan informasi artikel hilang.

**Decoder**

```text
Input decoder (token sebelumnya)
        ↓
Embedding(vocab_size, embedding_dim)
        ↓
SimpleRNN(hidden_dim, return_sequences=True, initial_state=h_T)
        ↓
Dense(vocab_size) → logits
```

**Seq2Seq**

`Seq2Seq` menggabungkan Encoder dan Decoder. Saat training digunakan **teacher forcing**:

```text
summary_ids = [<SOS>, t1, t2, ..., <EOS>]
decoder_input  = [<SOS>, t1, t2, ..., t_{n-1}]
decoder_target = [t1, t2, ..., t_{n-1}, <EOS>]
```

Loss adalah cross-entropy per token yang mengabaikan posisi `<PAD>`. Saat inference, decoder berjalan **autoregressive** (greedy decoding): mulai dari `<SOS>`, prediksi token berikutnya, umpan balik token tersebut beserta hidden state, lalu berhenti saat `<EOS>` atau batas panjang tercapai.

#### 3.2.3 Konfigurasi Eksperimen

Semua konfigurasi memakai embedding dimension 128, batch size 64, 8 epoch, dropout 0.2, gradient clipping 5.0, dan seed 42.

| Kode | Optimizer | Learning rate | Embedding | Hidden | Keterangan |
|---|---|---:|---:|---:|---|
| `adam_lr0.001_h128` | Adam | 0.001 | 128 | 128 | Baseline |
| `rmsprop_lr0.001_h128` | RMSprop | 0.001 | 128 | 128 | Perbandingan optimizer |
| `adam_lr0.0001_h128` | Adam | 0.0001 | 128 | 128 | Learning rate lebih kecil |
| `adam_lr0.001_h256` | Adam | 0.001 | 128 | 256 | Hidden dimension lebih besar |

Jumlah parameter:

- Konfigurasi hidden 128: **4.601.477 parameter**
- Konfigurasi hidden 256: **6.240.773 parameter**

#### 3.2.4 Prosedur Training

- Loss: `MaskedSparseCategoricalCrossentropy(from_logits=True)`
- Optimizer: Adam / RMSprop, `clipnorm=5.0`
- Metrik: masked token accuracy
- Epoch: 8
- Batch size: 64
- Validation set dipakai untuk memilih bobot terbaik (`ModelCheckpoint` pada `val_loss`).
- Waktu eksekusi setiap epoch dicatat oleh callback `EpochTimer`.
- Inference evaluasi: greedy decoding, maksimum 25 langkah.
- Evaluasi test: 618 sampel.

#### 3.2.5 Evaluasi ROUGE

Implementasi ROUGE berada di `summarize/rouge.py` dan tidak membutuhkan dependensi eksternal.

- **ROUGE-1:** overlap unigram dengan clipping.
- **ROUGE-2:** overlap bigram dengan clipping.
- **ROUGE-L:** Longest Common Subsequence (LCS).
- Skor yang dilaporkan adalah **rata-rata F1 per sampel** (macro average) pada 618 sampel uji.

### 3.3 Cara Menjalankan Seluruh Eksperimen

Perintah klasifikasi dijalankan dari direktori induk yang memuat folder `rnn`, yaitu `D:\notebook-playground`. Perintah summarization dijalankan dari dalam folder `rnn/summarize` karena seluruh skripnya memakai import lokal (`config`, `data`, `model`, dan seterusnya). Path data dan output tetap diturunkan dari lokasi masing-masing skrip, sehingga tidak bergantung pada direktori kerja.

**Klasifikasi sentimen**

```bash
cd D:\notebook-playground
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

# Eksperimen utama
python rnn/klasifikasi/klasifikasi_sentimen_rnn.py

# Eksperimen pembanding dan tuning (23 konfigurasi x 3 seed)
python -m rnn.klasifikasi.eksperimen_tuning
```

Skrip tuning akan memakai kembali hasil yang sudah ada di `rnn/klasifikasi/output_rnn/tuning/hasil_tuning_per_seed.csv` dan hanya melatih run yang belum tersedia. Jika ingin mengulang dari awal, hapus folder `rnn/klasifikasi/output_rnn/tuning/` terlebih dahulu.

**Text summarization**

```bash
cd D:\notebook-playground\rnn\summarize
pip install -r requirements.txt

# Eksperimen utama (4 konfigurasi)
python experiments.py --max-samples 6000 --epochs 8 --clean

# Seluruh dataset (lebih lama)
python experiments.py --max-samples 0 --epochs 10

# Evaluasi dan tampilkan tabel
python evaluate.py

# Inference satu teks
python summarize.py \
  --experiment outputs/best_model \
  --text "Fokuslah pada hal positif pasangan Anda dan luangkan waktu untuk menghargai hal-hal kecil."
```

Alternatif tanpa berpindah direktori, dari `D:\notebook-playground\rnn`:

```bash
python summarize/experiments.py --max-samples 6000 --epochs 8 --clean
python summarize/evaluate.py
python summarize/summarize.py --experiment summarize/outputs/best_model --text "..."
```

Atau dari Python (dijalankan dari dalam folder `rnn/summarize`):

```python
from summarize import summarize

ringkasan = summarize(
    "Fokuslah pada hal positif pasangan Anda dan luangkan waktu untuk menghargai hal-hal kecil.",
    experiment_dir="outputs/best_model",
)
print(ringkasan)
```

---

## 4. Hasil dan Analisis

### 4.1 Klasifikasi Sentimen

#### 4.1.1 Kurva Loss dan Accuracy

![Kurva Loss dan Accuracy selama training](klasifikasi/output_rnn/kurva_loss_accuracy.png)

*Gambar 1. Kurva training dan validasi model SimpleRNN.*

Beberapa pola penting dari kurva training:

- **Loss data latih** menurun dari 0,7211 pada epoch pertama menjadi 0,2061 pada epoch terakhir. **Accuracy data latih** meningkat dari 49,62% menjadi 96,24%.
- **Loss validasi** menurun dari 0,6986 menjadi 0,5350, dengan nilai minimum 0,5132 pada epoch ke-27. **Accuracy validasi** sempat mencapai 83,33% pada epoch 24–26, lalu berfluktuasi dan berada di 79,17% pada epoch terakhir.
- Jarak antara performa data latih dan validasi semakin lebar setelah pertengahan training. Data latih mendekati 100%, sedangkan validasi tertahan di kisaran 75%–83%. Pola ini menunjukkan **overfitting**, yaitu model mulai menghafal data latih dan kurang mampu melakukan generalisasi.
- Regularisasi berupa dropout, recurrent dropout, dan L2 memperlambat overfitting. Early stopping kemudian memilih model dengan `val_loss` terbaik, bukan model pada epoch terakhir yang sudah terlalu menyesuaikan diri dengan data latih.

Pemilihan model berdasarkan `val_loss` menghasilkan model pada epoch ke-27. Pada epoch tersebut, accuracy data latih adalah 94,74% dan accuracy validasi 75,00%. Walaupun accuracy validasi bukan yang tertinggi, `val_loss` yang minimum menunjukkan model tersebut memberikan probabilitas prediksi yang relatif lebih terkalibrasi dibandingkan epoch-epoch setelahnya.

#### 4.1.2 Confusion Matrix

![Confusion Matrix pada data uji](klasifikasi/output_rnn/confusion_matrix.png)

*Gambar 2. Confusion matrix model pada 40 data uji.*

| Aktual \ Prediksi | Negative | Positive | Total |
|---|---:|---:|---:|
| **Negative** | 15 (TN) | 5 (FP) | 20 |
| **Positive** | 4 (FN) | 16 (TP) | 20 |
| **Total** | 19 | 21 | 40 |

Keterangan:

- **TN (True Negative)** = 15: tweet negatif yang benar diprediksi negatif.
- **FP (False Positive)** = 5: tweet negatif yang salah diprediksi positif.
- **FN (False Negative)** = 4: tweet positif yang salah diprediksi negatif.
- **TP (True Positive)** = 16: tweet positif yang benar diprediksi positif.

Model membuat 9 kesalahan dari 40 data uji, terdiri atas 5 false positive dan 4 false negative. Distribusi kesalahan cukup seimbang, sehingga model tidak condong secara ekstrem ke salah satu kelas. Prediksi positif sedikit lebih banyak (21) dibandingkan prediksi negatif (19), tetapi selisihnya kecil.

#### 4.1.3 Hasil Metrik Evaluasi

**Metrik keseluruhan**

| Metrik | Nilai |
|---|---:|
| Accuracy | **0,7750 (77,50%)** |
| Precision (weighted) | **0,7757** |
| Recall (weighted) | **0,7750** |
| F1-score (weighted) | **0,7749** |
| Precision (macro) | 0,7757 |
| Recall (macro) | 0,7750 |
| F1-score (macro) | 0,7749 |

**Metrik per kelas**

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0,7895 | 0,7500 | 0,7692 | 20 |
| positive | 0,7619 | 0,8000 | 0,7805 | 20 |
| **Macro average** | **0,7757** | **0,7750** | **0,7749** | **40** |
| **Weighted average** | **0,7757** | **0,7750** | **0,7749** | **40** |

**Pembacaan hasil**

- **Accuracy 77,50%** berarti model benar memprediksi 31 dari 40 tweet.
- Karena data uji seimbang 20 negatif dan 20 positif, baseline model yang selalu menebak satu kelas adalah 50%. Model meningkatkan accuracy sebesar **27,5 poin persentase** di atas baseline tersebut.
- **Precision negatif** (0,7895) lebih tinggi daripada precision positif (0,7619). Artinya, ketika model memprediksi negatif, peluang prediksi tersebut benar sedikit lebih besar.
- **Recall positif** (0,8000) lebih tinggi daripada recall negatif (0,7500). Model lebih berhasil menemukan tweet positif daripada tweet negatif.
- **F1-score** kedua kelas relatif seimbang, yaitu 0,7692 untuk negatif dan 0,7805 untuk positif. Tidak ada kelas yang tertinggal secara ekstrem.

Dengan hanya 40 data uji, satu kesalahan mengubah accuracy sebesar 2,5 poin persentase. Interval kepercayaan 95% untuk accuracy (metode Clopper–Pearson) berada di kisaran **61,6%–89,2%**. Rentang yang lebar ini menunjukkan bahwa hasil evaluasi perlu dipandang sebagai estimasi, bukan angka performa yang pasti.

#### 4.1.4 Hasil Eksperimen Pembanding dan Tuning Hyperparameter

Sebanyak 23 konfigurasi unik diuji dengan tiga random seed, sehingga totalnya 69 run training. Semua konfigurasi menggunakan split data dan vektorisasi yang sama. Pemilihan konfigurasi dilakukan berdasarkan rata-rata validation F1-score; data uji baru digunakan pada evaluasi akhir konfigurasi terbaik. Tabel lengkap tersedia di `klasifikasi/output_rnn/tuning/ringkasan_tuning.csv`, sedangkan hasil per run ada di `klasifikasi/output_rnn/tuning/hasil_tuning_per_seed.csv`.

Sebagai pembanding awal, konfigurasi baseline SimpleRNN 32 unit memperoleh validation accuracy **0,6250 ± 0,1909**, validation F1-score **0,6237 ± 0,1917**, dan validation loss **0,6358 ± 0,1063** dari tiga seed. Variasi antar seed yang sangat lebar menunjukkan bahwa SimpleRNN cukup sensitif terhadap inisialisasi pada dataset kecil ini.

##### 4.1.4.1 Perbandingan optimizer

![Perbandingan optimizer](klasifikasi/output_rnn/tuning/perbandingan_optimizer.png)

*Gambar 3. Validation accuracy dan F1-score untuk empat optimizer.*

| Optimizer | Validation Accuracy | Validation F1-score | Validation Loss | Rata-rata waktu training (detik) |
|---|---:|---:|---:|---:|
| Adam | 0,6250 ± 0,1909 | 0,6237 ± 0,1917 | 0,6358 ± 0,1063 | 3,7 |
| AdamW | 0,6250 ± 0,1909 | 0,6237 ± 0,1917 | 0,6358 ± 0,1063 | 3,7 |
| RMSprop | **0,6806 ± 0,1684** | **0,6804 ± 0,1684** | **0,6016 ± 0,0855** | 5,1 |
| SGD | 0,5139 ± 0,0481 | 0,3846 ± 0,0609 | 0,6899 ± 0,0071 | 3,0 |

RMSprop memiliki rata-rata validation F1-score tertinggi, tetapi simpangan bakunya juga besar (0,1684). Selisih rata-rata terhadap Adam hanya sekitar 0,057 dan masih jauh lebih kecil daripada variasi antar seed, sehingga belum cukup bukti bahwa RMSprop secara konsisten lebih baik. Adam dan AdamW menghasilkan nilai yang identik pada presisi empat desimal karena weight decay AdamW yang digunakan cukup kecil (1e-4) dan early stopping berhenti pada titik yang sama.

SGD dengan learning rate 5e-3 dan momentum 0.9 justru tertinggal jauh. Validation F1-score-nya hanya 0,3846 dan loss-nya paling tinggi, yang mengindikasikan model belum cukup terlatih dalam batas 60 epoch. Hasil ini menunjukkan bahwa pemilihan optimizer perlu disertai penyesuaian learning rate dan jumlah epoch; perbandingan optimizer tanpa kontrol tersebut tidak dapat disimpulkan secara sederhana.

##### 4.1.4.2 Perbandingan arsitektur RNN

![Perbandingan arsitektur RNN](klasifikasi/output_rnn/tuning/perbandingan_arsitektur.png)

*Gambar 4. Validation accuracy dan F1-score untuk enam varian arsitektur RNN.*

| Arsitektur | Validation Accuracy | Validation F1-score | Validation Loss | Rata-rata waktu training (detik) |
|---|---:|---:|---:|---:|
| SimpleRNN | 0,6250 ± 0,1909 | 0,6237 ± 0,1917 | 0,6358 ± 0,1063 | 3,7 |
| **GRU** | **0,8611 ± 0,0241** | **0,8610 ± 0,0239** | **0,4584 ± 0,0055** | 5,8 |
| LSTM | **0,8611 ± 0,0241** | 0,8606 ± 0,0246 | 0,5485 ± 0,0279 | 5,3 |
| BiSimpleRNN | 0,5694 ± 0,1879 | 0,5628 ± 0,1895 | 0,6928 ± 0,0193 | 5,3 |
| BiGRU | 0,7500 ± 0,1102 | 0,7383 ± 0,1300 | 0,5288 ± 0,0529 | 9,2 |
| BiLSTM | 0,7639 ± 0,0481 | 0,7573 ± 0,0538 | 0,6379 ± 0,0204 | 9,2 |

Perbedaan paling jelas datang dari **jenis arsitektur**, bukan dari optimizer atau hyperparameter lain. GRU dan LSTM sama-sama mencapai rata-rata validation accuracy 0,8611 dan F1-score sekitar 0,861, jauh di atas SimpleRNN (0,6237). Yang lebih penting, simpangan bakunya mengecil drastis: dari 0,1917 pada SimpleRNN menjadi sekitar 0,024 pada GRU/LSTM. Artinya, GRU dan LSTM tidak hanya lebih akurat secara rata-rata, tetapi juga jauh lebih stabil pada tiga seed yang diuji.

GRU sedikit lebih unggul dari LSTM pada validation loss (0,4584 vs 0,5485), sedangkan waktu training keduanya sebanding (5,8 vs 5,3 detik; selisih kecil pada data sekecil ini). Struktur gating pada GRU/LSTM membantu model mengatur informasi mana yang perlu dipertahankan, sehingga lebih tahan terhadap *vanishing gradient* dibandingkan SimpleRNN.

Sebaliknya, varian bidirectional tidak memberikan peningkatan. BiGRU dan BiLSTM berada di bawah versi satu arahnya, sedangkan BiSimpleRNN justru paling rendah. Dengan hanya 133 data latih dan tweet yang relatif pendek (23 token), menambah arah pemrosesan berarti menambah parameter dan risiko overfitting tanpa cukup data untuk memanfaatkannya. Waktu training varian bidirectional juga sekitar 1,6–1,7 kali lipat dibanding versi satu arah.

##### 4.1.4.3 Tuning hyperparameter

![Pengaruh hyperparameter](klasifikasi/output_rnn/tuning/tuning_hyperparameter.png)

*Gambar 5. Pengaruh tiap hyperparameter terhadap validation F1-score (mean ± std, 3 seed). Garis putus-putus adalah baseline.*

| Kelompok | Nilai terbaik | Validation F1-score | Selisih terhadap baseline |
|---|---|---:|---:|
| Jumlah unit RNN | 16 | 0,6747 | +0,0510 |
| Dimensi embedding | 64 | 0,6237 | 0,0000 |
| Dropout RNN | 0.7 | 0,6377 | +0,0140 |
| Recurrent dropout | 0.0 | 0,6539 | +0,0302 |
| Batch size | 16 | 0,6237 | 0,0000 |
| Learning rate | 5e-4 | 0,6237 | 0,0000 |
| L2 embedding | 1e-4 | 0,6237 | 0,0000 |

Tabel di atas berasal dari tuning satu faktor pada satu waktu dengan arsitektur SimpleRNN. Nilai “terbaik” pada tiap baris belum tentu tetap terbaik jika digabungkan, sehingga konfigurasi terbaik keseluruhan tetap ditentukan dari 23 konfigurasi yang diuji, bukan dari menggabungkan nilai terbaik per faktor secara langsung. Konfigurasi terbaik keseluruhan ternyata adalah GRU 32 unit, bukan SimpleRNN dengan unit 16.

Tidak ada satu pun perubahan hyperparameter yang menghasilkan peningkatan besar dan konsisten. Selisih terbesar hanya sekitar 0,051 (unit 16 vs 32), sementara simpangan baku antar seed pada kelompok yang sama berkisar 0,08–0,22. Dengan kata lain, perbedaan tersebut **berada di dalam noise** dan belum dapat dianggap sebagai peningkatan yang nyata.

Beberapa pola yang tetap dapat dibaca:

- **Unit 16** sedikit lebih baik daripada 32 dan 64. Model yang lebih kecil lebih sesuai untuk 133 data latih; unit 64 tidak memberikan peningkatan yang berarti dibandingkan unit 32.
- **Embedding 32** paling buruk (F1 0,4853), sedangkan 64 dan 128 hampir setara. Representasi 32 dimensi terlalu sempit untuk 845 token, tetapi memperbesar ke 128 tidak lagi memberi manfaat.
- **Dropout 0.7** sedikit lebih tinggi dari 0.5 dan 0.3, tetapi perbedaannya kecil. Recurrent dropout 0.0 justru sedikit lebih baik dari 0.2 dan 0.4; regularisasi rekuren yang terlalu kuat dapat mengganggu pembelajaran pada sequence pendek.
- **Batch size 8** lebih buruk daripada 16/32. Batch 16 dan 32 hampir identik; batch 16 dipilih karena memberi update lebih sering tanpa terlalu banyak noise.
- **Learning rate 1e-4** membuat model underfit (F1 0,5341), sedangkan 1e-3 tidak lebih baik dari 5e-4. Learning rate 5e-4 adalah titik yang paling seimbang.
- **L2 1e-3** sedikit lebih rendah daripada 1e-4, tetapi masih lebih tinggi daripada tanpa L2 (0). Regularisasi L2 yang terlalu kuat tidak diperlukan karena dropout sudah cukup menahan overfitting.

##### 4.1.4.4 Konfigurasi terbaik dan evaluasi data uji

Konfigurasi dengan rata-rata validation F1-score tertinggi adalah **GRU 32 unit** dengan hyperparameter baseline lainnya:

| Parameter | Nilai |
|---|---|
| Arsitektur | GRU, satu arah |
| Jumlah unit | 32 |
| Dimensi embedding | 64 |
| Dropout RNN | 0.5 |
| Recurrent dropout | 0.2 |
| Dense hidden | 32 |
| Dropout dense | 0.5 |
| L2 embedding | 1e-4 |
| Learning rate | 5e-4 |
| Batch size | 16 |
| Optimizer | Adam |
| Jumlah parameter | 64.577 |

Konfigurasi ini memperoleh validation accuracy **0,8611 ± 0,0241**, validation F1-score **0,8610 ± 0,0239**, dan validation loss **0,4584 ± 0,0055**. Model kemudian dilatih ulang pada data latih dengan seed 42 dan dievaluasi pada 40 data uji.

![Kurva training konfigurasi terbaik](klasifikasi/output_rnn/tuning/model_terbaik/kurva_model_tuning_terbaik.png)

*Gambar 6. Kurva training konfigurasi GRU terbaik. Early stopping memilih epoch ke-21.*

![Confusion matrix konfigurasi terbaik](klasifikasi/output_rnn/tuning/model_terbaik/confusion_matrix.png)

*Gambar 7. Confusion matrix konfigurasi GRU terbaik pada data uji.*

| Aktual \ Prediksi | Negative | Positive | Total |
|---|---:|---:|---:|
| **Negative** | 15 (TN) | 5 (FP) | 20 |
| **Positive** | 6 (FN) | 14 (TP) | 20 |
| **Total** | 21 | 19 | 40 |

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|---|---:|---:|---:|---:|
| Baseline SimpleRNN | **0,7750** | **0,7757** | **0,7750** | **0,7749** |
| Tuning terbaik (GRU) | 0,7250 | 0,7256 | 0,7250 | 0,7248 |

Tabel perbandingan ini juga tersimpan di `klasifikasi/output_rnn/tuning/model_terbaik/perbandingan_baseline_vs_terbaik.csv`.

Hasil data uji konfigurasi GRU adalah **72,50%** (29 dari 40 benar), dengan 15 true negative, 5 false positive, 6 false negative, dan 14 true positive. Angka ini **lebih rendah 5 poin persentase** daripada baseline SimpleRNN (77,50%). Perbedaannya setara dengan dua tweet saja. Interval kepercayaan 95% accuracy GRU adalah sekitar **56,1%–85,4%**, sedangkan baseline **61,6%–89,2%**; kedua interval saling tumpang tindih lebar. Oleh karena itu, perbedaan 72,50% vs 77,50% **belum dapat dianggap sebagai perbedaan yang signifikan** pada data uji sekecil ini.

Kurva training GRU juga menunjukkan overfitting: training accuracy mencapai 100%, sedangkan validation accuracy tertahan di sekitar 83%. Early stopping memilih epoch ke-21 dengan validation loss terbaik. Pola ini konsisten dengan keterbatasan data, bukan berarti GRU lebih buruk secara struktural.

##### 4.1.4.5 Ringkasan temuan tuning

1. **Arsitektur adalah faktor yang paling berpengaruh.** GRU/LSTM meningkatkan rata-rata validation F1 dari 0,6237 menjadi sekitar 0,861 dan menurunkan simpangan baku antar seed dari 0,192 menjadi 0,024.
2. **Bidirectional tidak membantu pada dataset ini.** BiGRU/BiLSTM lebih rendah dari versi satu arah dan membutuhkan waktu training hampir dua kali lipat.
3. **Perubahan hyperparameter tunggal tidak konklusif.** Semua selisih berada dalam rentang noise validation set yang hanya 24 sampel; tuning satu faktor pada satu waktu belum cukup untuk menemukan kombinasi optimal.
4. **RMSprop sedikit lebih tinggi secara rata-rata, tetapi variansnya besar.** Adam/AdamW tetap pilihan yang aman; SGD underfit pada konfigurasi yang diuji.
5. **GRU lebih stabil pada validasi, tetapi belum mengungguli baseline pada data uji.** Baseline 77,50% vs GRU 72,50% hanya berbeda dua tweet dan interval kepercayaannya tumpang tindih. Diperlukan k-fold cross-validation atau data yang lebih banyak sebelum menyimpulkan pemenang secara definitif.

#### 4.1.5 Pembahasan Klasifikasi Sentimen

**Hal yang sudah bekerja dengan baik**

1. **Alur eksperimen valid.** Data uji dipisahkan sejak awal, vektorisasi hanya di-*adapt* pada data latih, dan data validasi berasal dari data latih. Eksperimen tuning juga memakai split yang sama dan memilih konfigurasi hanya berdasarkan validation set; data uji tidak digunakan untuk memilih model.
2. **Model mampu mempelajari sinyal sentimen.** Accuracy model utama 77,50% jauh di atas baseline 50%, menunjukkan bahwa model menangkap kata-kata yang berkaitan dengan opini, seperti `bagus`, `keren`, `jelek`, `kecewa`, dan `membosankan`.
3. **Preprocessing dasar cukup efektif.** Penghapusan mention, URL, dan karakter khusus mengurangi noise tanpa menghilangkan kata utama yang membawa sentimen.
4. **Regularisasi dan early stopping membantu.** Tanpa regularisasi, SimpleRNN dengan dataset sekecil ini sangat cepat overfit. Dengan dropout, recurrent dropout, L2, dan pemilihan bobot terbaik, model masih menghasilkan performa yang wajar pada data uji.
5. **Tuning memberikan informasi desain yang berguna.** GRU/LSTM secara konsisten lebih baik daripada SimpleRNN pada validation set, dengan rata-rata F1 sekitar 0,861 dan simpangan baku yang jauh lebih kecil. Temuan ini lebih informatif daripada perbedaan kecil antarseed pada model baseline.

**Sumber kesalahan dan keterbatasan**

1. **Overfitting akibat data terbatas.** Data latih hanya 133 tweet. Training accuracy model utama mencapai 96,24%, sedangkan pada GRU tuning bahkan mencapai 100%; validation accuracy tertahan di kisaran 75%–83%. Model memiliki kapasitas yang cukup untuk menghafal pola data latih.
2. **Keterbatasan SimpleRNN.** SimpleRNN lebih rentan terhadap *vanishing gradient* dibandingkan LSTM atau GRU. Tuning mengonfirmasi hal ini: rata-rata validation F1 SimpleRNN hanya 0,6237 dengan simpangan baku 0,1917, sedangkan GRU mencapai 0,8610 dengan simpangan baku 0,0239.
3. **Preprocessing masih dasar.** Tidak ada stemming, normalisasi kata gaul, penghapusan stopword, atau penanganan negasi. Kata seperti `gak`, `tidak`, dan `nggak` dapat menjadi token terpisah; kata seperti `bgt` dan `yg` juga tidak dinormalisasi. Hal ini memperbesar kosakata dan mempersulit generalisasi.
4. **Kehilangan sebagian informasi.** Penghapusan angka, tanda baca, dan emoji menghilangkan isyarat sentimen tertentu. Contohnya, emoji sedih atau tanda seru dapat memperkuat opini negatif, tetapi tidak lagi tersedia untuk model.
5. **Sarkasme dan sentimen campuran.** Tweet seperti pujian yang disertai kritik, atau kalimat sarkastik, sulit diklasifikasikan oleh model berbasis embedding kata yang dilatih dari data kecil.
6. **Evaluasi bergantung pada satu split.** Hasil 77,50% berasal dari satu pembagian data dengan `random_state=42`. Tiga random seed pada eksperimen tuning hanya mengukur variasi inisialisasi dan urutan training, bukan variasi pembagian data. Pembagian lain dapat menghasilkan angka yang berbeda cukup besar karena ukuran data uji hanya 40.
7. **Validation set terlalu kecil untuk tuning yang rinci.** Hanya ada 24 data validasi, sehingga satu prediksi yang berubah setara 4,17 poin persentase. Perbedaan hyperparameter di bawah sekitar 0,05 F1 pada eksperimen ini belum dapat dipercaya sebagai peningkatan yang nyata.

### 4.2 Text Summarization

#### 4.2.1 Training Loss dan Validation Loss

Tabel ringkasan hasil akhir:

| Konfigurasi | Train Loss | Val Loss | Best Val Loss | Training Time (s) | Avg Epoch Time (s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam_lr0.001_h128` | 5,5090 | 6,5027 | **6,4938** | 104,92 | 13,02 | 0,0480 | 0,0003 | 0,0476 |
| `rmsprop_lr0.001_h128` | 6,2670 | 6,5372 | 6,5372 | **100,77** | **12,51** | **0,0707** | 0,0000 | **0,0696** |
| `adam_lr0.0001_h128` | 6,5090 | 6,8797 | 6,8797 | 107,86 | 13,38 | 0,0000 | 0,0000 | 0,0000 |
| `adam_lr0.001_h256` | **5,2133** | 6,6066 | 6,5435 | 158,98 | 19,80 | 0,0108 | 0,0000 | 0,0108 |

> Waktu pada kolom **Training Time** adalah durasi `model.fit` (training + validasi) dan tidak termasuk evaluasi ROUGE pada test set.

Grafik training dan validation loss:

![Training dan Validation Loss](summarize/outputs/loss_curves.png)

*Gambar 8. Kurva training dan validation loss untuk empat konfigurasi Seq2Seq.*

**Observasi:**

1. **Adam lr 0.001, hidden 128** memberikan validation loss terendah (6,4938) dan penurunan training loss yang stabil.
2. **Adam lr 0.001, hidden 256** memiliki training loss paling rendah (5,2133), tetapi validation loss lebih tinggi (6,6066) dan mulai meningkat setelah epoch 6. Ini indikasi **overfitting**.
3. **Adam lr 0.0001** masih berada pada fase underfitting: validation loss 6,8797 dan masih menurun perlahan. Model ini bahkan gagal menghasilkan token (langsung memprediksi `<EOS>`), sehingga ROUGE = 0.
4. **RMSprop lr 0.001** memiliki training loss lebih tinggi daripada Adam, tetapi validation loss kompetitif (6,5372) dan menghasilkan ROUGE-L tertinggi.
5. Semua kurva validation loss mendatar di sekitar 6,5. Model belum mampu mempelajari pemetaan artikel→ringkasan secara spesifik.

#### 4.2.2 Execution Time per Epoch

| Konfigurasi | Rata-rata per epoch (s) | Total 8 epoch (s) | Rasio terhadap baseline |
|---|---:|---:|---:|
| `rmsprop_lr0.001_h128` | 12,51 | 100,77 | 0,96× |
| `adam_lr0.001_h128` | 13,02 | 104,92 | 1,00× |
| `adam_lr0.0001_h128` | 13,38 | 107,86 | 1,03× |
| `adam_lr0.001_h256` | 19,80 | 158,98 | **1,52×** |

Grafik waktu eksekusi per epoch:

![Execution Time per Epoch](summarize/outputs/epoch_times.png)

*Gambar 9. Waktu eksekusi per epoch untuk empat konfigurasi.*

**Observasi:**

- Hidden dimension 256 meningkatkan waktu per epoch sekitar **52%** dibanding hidden 128, tetapi tidak meningkatkan ROUGE secara konsisten.
- RMSprop sedikit lebih cepat daripada Adam pada konfigurasi yang sama.
- Waktu per epoch relatif stabil setelah epoch pertama (epoch pertama lebih lambat karena proses build/kompilasi).
- Tabel lengkap per epoch tersedia di `summarize/outputs/per_epoch_epoch_time_sec.csv` dan `summarize/outputs/per_epoch_epoch_time_sec.md`.

#### 4.2.3 Hasil ROUGE pada Test Set

| Konfigurasi | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| `adam_lr0.001_h128` | 0,0480 | 0,0003 | 0,0476 |
| `rmsprop_lr0.001_h128` | **0,0707** | 0,0000 | **0,0696** |
| `adam_lr0.0001_h128` | 0,0000 | 0,0000 | 0,0000 |
| `adam_lr0.001_h256` | 0,0108 | 0,0000 | 0,0108 |

Grafik perbandingan ROUGE:

![Perbandingan ROUGE](summarize/outputs/rouge_comparison.png)

*Gambar 10. Perbandingan ROUGE-1, ROUGE-2, dan ROUGE-L antar konfigurasi.*

**Contoh prediksi pada test set:**

| Referensi | Prediksi `adam_lr0.001_h128` | Prediksi `rmsprop_lr0.001_h128` | Prediksi `adam_lr0.001_h256` |
|---|---|---|---|
| hitunglah laju metabolisme basal lmb anda | gunakan bahwa anda | gunakan dengan yang | gunakan tanda koma |
| gunakan kalkulator lmb daring | gunakan bahwa anda | gunakan dengan yang | gunakan tanda koma |
| mulailah menulis jurnal makanan | gunakan bahwa anda | gunakan dengan yang | gunakan tanda koma |

**Temuan penting:**

- Setiap model menghasilkan **satu prediksi yang sama untuk seluruh 618 sampel uji** (unique prediction = 1). Ini menunjukkan **mode collapse**: decoder berperilaku seperti language model umum, bukan model kondisional terhadap artikel.
- `adam_lr0.0001_h128` menghasilkan **618 prediksi kosong** karena selalu memprediksi `<EOS>` pada langkah pertama.
- ROUGE-2 hampir nol karena frasa generik yang dihasilkan tidak memiliki bigram yang sama dengan referensi.
- RMSprop menang ROUGE-L bukan karena ringkasannya lebih baik secara semantik, tetapi karena frasa `"gunakan dengan yang"` kebetulan memiliki lebih banyak unigram umum yang juga muncul di referensi.

#### 4.2.4 Diskusi Text Summarization

Hasil ini memperlihatkan karakteristik Vanilla RNN Seq2Seq pada tugas summarization:

1. **Bottleneck satu vektor hidden state.** Seluruh artikel (hingga 128 token) harus diringkas menjadi satu vektor. Untuk artikel yang panjang, informasi awal cepat hilang.
2. **Vanishing gradient.** RNN sederhana sulit mempertahankan informasi jangka panjang, sehingga decoder cenderung mengandalkan token terakhir dan frekuensi token pada data latih.
3. **Frequency bias pada cross-entropy.** Token/frasa umum seperti “gunakan” mendominasi loss. Model menemukan solusi lokal berupa frasa generik yang aman secara loss, tetapi tidak informatif.
4. **Exposure bias.** Saat training decoder selalu menerima token ground-truth (teacher forcing), sedangkan saat inference menerima prediksinya sendiri. Kesalahan kecil di awal dapat membuat decoder keluar dari distribusi training.
5. **Data dan durasi terbatas.** Eksperimen hanya memakai 6.011 pasangan dan 8 epoch pada CPU. Dataset penuh (365.121 pasangan) dan epoch lebih banyak berpotensi memperbaiki hasil, tetapi tetap dibatasi oleh kapasitas Vanilla RNN.
6. **Validation loss tidak selalu sejalan dengan ROUGE.** Adam hidden 128 memiliki validation loss terbaik, tetapi ROUGE-L lebih rendah daripada RMSprop. Hal ini menunjukkan bahwa loss token-level tidak identik dengan kualitas ringkasan.

Secara implementasi, pipeline sudah berjalan benar: training loss menurun, validation loss terpantau, inference autoregressive menghasilkan sequence, dan ROUGE dapat dihitung. Namun, kualitas ringkasan yang dihasilkan masih jauh dari memadai untuk penggunaan nyata.

#### 4.2.5 Keterbatasan Text Summarization

- Hanya menggunakan 6.011 dari 365.121 pasangan data.
- Belum menggunakan attention mechanism.
- Belum menggunakan beam search, scheduled sampling, atau coverage mechanism.
- Tokenisasi masih sederhana; belum ada stemming/lemmatization Bahasa Indonesia.
- Evaluasi ROUGE berbasis overlap kata, belum menangkap kualitas semantik.
- Model dijalankan di CPU, sehingga eksplorasi hyperparameter dan epoch terbatas.

### 4.3 Perbandingan Lintas Eksperimen

| Aspek | Klasifikasi Sentimen | Text Summarization |
|---|---|---|
| Bentuk tugas | Many-to-one | Many-to-many (Seq2Seq) |
| Ukuran data | 197 tweet (133 latih / 24 validasi / 40 uji) | 6.011 pasangan (4.793 / 600 / 618) |
| Model utama | SimpleRNN 32 unit, 58.273 parameter | Seq2Seq SimpleRNN, 4.601.477–6.240.773 parameter |
| Durasi training | Maksimum 60 epoch, berhenti pada epoch 36 | 8 epoch |
| Metrik utama | Accuracy 77,50%; F1 weighted 0,7749 | ROUGE-1 0,0707; ROUGE-L 0,0696 |
| Hasil validasi terbaik | F1 0,8610 (GRU 32 unit) | Validation loss 6,4938 (Adam lr 0.001 hidden 128) |
| Masalah utama | Overfitting, data kecil, variansi antar seed besar | Mode collapse, bottleneck hidden state, vanishing gradient |
| Perbandingan arsitektur | GRU/LSTM jauh lebih stabil daripada SimpleRNN | Belum dilakukan; merupakan keterbatasan |
| Status hasil | Layak sebagai baseline dan menunjukkan sinyal sentimen | Belum memadai untuk penggunaan nyata |

Beberapa temuan lintas eksperimen:

1. **Kompleksitas tugas menentukan batas Vanilla RNN.** Pada klasifikasi many-to-one dengan sequence pendek (23 token), SimpleRNN masih mampu mencapai 77,50%. Pada summarization many-to-many dengan artikel hingga 128 token, model gagal mempertahankan informasi dan mengalami mode collapse.
2. **Ukuran data sangat memengaruhi stabilitas.** Klasifikasi dengan 133 data latih menunjukkan variansi antar seed yang besar pada SimpleRNN (simpangan baku F1 0,1917). Summarization dengan 4.793 data latih tetap mengalami mode collapse karena keterbatasan kapasitas model dan panjang sequence.
3. **Gating membantu, tetapi bukan jaminan performa uji.** GRU/LSTM jelas lebih stabil pada validation set klasifikasi, tetapi konfigurasi GRU terbaik belum mengungguli baseline pada 40 data uji. Perbedaan dua tweet belum cukup untuk menyimpulkan keunggulan.
4. **Metrik task-level dan loss tidak selalu sejalan.** Pada summarization, validation loss terbaik tidak menghasilkan ROUGE terbaik. Pada klasifikasi, validation loss terbaik juga tidak selalu menghasilkan accuracy validasi tertinggi. Pemilihan model perlu mempertimbangkan metrik yang relevan dengan tugas.
5. **Evaluasi pada data kecil menghasilkan ketidakpastian besar.** Interval kepercayaan accuracy klasifikasi melebar (61,6%–89,2% untuk baseline dan 56,1%–85,4% untuk GRU). Pada summarization, 618 sampel uji lebih banyak, tetapi mode collapse membuat perbedaan ROUGE antar konfigurasi tidak bermakna secara semantik.
6. **Pipeline eksperimen sudah modular dan reproducible.** Kedua eksperimen memisahkan preprocessing, model, training, evaluasi, dan inference; menyimpan artefak; serta menggunakan seed tetap. Struktur ini memudahkan pengembangan lanjutan, misalnya penambahan attention atau penggantian sel RNN.

---

## 5. Kesimpulan

### 5.1 Klasifikasi Sentimen

Eksperimen klasifikasi sentimen tweet opini film menggunakan **SimpleRNN many-to-one** telah berhasil diimplementasikan dengan TensorFlow/Keras. Dataset lokal berisi 200 tweet, yang setelah pembersihan duplikat menjadi 197 tweet dengan komposisi 99 negatif dan 98 positif. Data dibagi secara stratified menjadi 133 data latih, 24 data validasi, dan 40 data uji.

Model menerima sequence token hasil `TextVectorization` dengan panjang 23 token dan kosakata 845 kata, kemudian memprosesnya melalui embedding, SimpleRNN 32 unit, layer dense, dropout, dan output sigmoid. Regularisasi L2, dropout, recurrent dropout, ReduceLROnPlateau, serta early stopping digunakan untuk mengendalikan overfitting pada dataset yang kecil.

Hasil evaluasi akhir pada data uji adalah:

| Metrik | Nilai |
|---|---:|
| Accuracy | **77,50%** |
| Precision (weighted) | **0,7757** |
| Recall (weighted) | **0,7750** |
| F1-score (weighted) | **0,7749** |

Model dengan benar memprediksi 31 dari 40 tweet. Confusion matrix menunjukkan 15 true negative, 5 false positive, 4 false negative, dan 16 true positive. Performa ini berada 27,5 poin persentase di atas baseline 50%, sehingga model terbukti mampu mempelajari pola sentimen dasar dari data.

Eksperimen pembanding menjalankan 23 konfigurasi unik dengan tiga random seed (69 run). Hasilnya, pemilihan arsitektur adalah faktor yang paling berpengaruh. GRU 32 unit memperoleh rata-rata validation accuracy **0,8611 ± 0,0241** dan validation F1-score **0,8610 ± 0,0239**, jauh lebih tinggi dan lebih stabil daripada baseline SimpleRNN (**0,6250 ± 0,1909** dan **0,6237 ± 0,1917**). LSTM memberikan hasil validation yang setara dengan GRU. Perubahan optimizer dan hyperparameter tunggal tidak menunjukkan peningkatan yang konsisten; sebagian besar selisih berada dalam noise validation set yang hanya 24 sampel.

Pada data uji, konfigurasi GRU terbaik memperoleh accuracy **72,50%** (29 dari 40), sedangkan baseline SimpleRNN **77,50%** (31 dari 40). Selisih 5 poin persentase tersebut setara dengan dua tweet, dan interval kepercayaan 95% kedua model saling tumpang tindih (GRU sekitar 56,1%–85,4%; baseline 61,6%–89,2%). Dengan demikian, belum cukup bukti untuk menyatakan salah satu model lebih unggul secara signifikan pada data uji ini. Baseline tetap menjadi model utama pada laporan, sedangkan GRU/LSTM menjadi kandidat yang lebih stabil untuk pengembangan selanjutnya.

Meskipun demikian, hasil tersebut masih dipengaruhi oleh keterbatasan jumlah data, overfitting, preprocessing yang baru mencapai tahap dasar, serta kemampuan SimpleRNN dalam menangkap konteks jangka panjang. Pengembangan lebih lanjut disarankan menggunakan validasi silang k-fold atau nested cross-validation, pretrained embedding, normalisasi kata gaul, dan penanganan negasi untuk meningkatkan kemampuan generalisasi model. GRU/LSTM dapat dipertimbangkan sebagai kandidat model karena lebih stabil pada validation set, tetapi keunggulannya pada data uji belum terbukti.

Seluruh ketentuan eksperimen klasifikasi telah dipenuhi:

- Dataset dibaca dari file lokal `dataset_tweet_sentiment_opini_film.csv`.
- Preprocessing teks dasar telah diterapkan.
- Dataset dibagi menjadi data latih, validasi, dan data uji.
- Kurva loss dan accuracy ditampilkan dan disimpan.
- Confusion matrix serta classification report (accuracy, precision, recall, F1-score) ditampilkan.
- Eksperimen pembanding 23 konfigurasi × 3 seed dan evaluasi konfigurasi terbaik telah dilakukan.
- Model terbaik dan seluruh artefak evaluasi disimpan di folder `klasifikasi/output_rnn/`.

### 5.2 Text Summarization

1. Implementasi Seq2Seq Vanilla RNN dengan TensorFlow/Keras berhasil dibuat secara modular, meliputi preprocessing, Vocabulary, Encoder, Decoder, Seq2Seq, training, inference, evaluasi ROUGE, dan pembuatan laporan otomatis.
2. Empat konfigurasi eksperimen berhasil dibandingkan:
   - Adam lr 0.001, hidden 128
   - RMSprop lr 0.001, hidden 128
   - Adam lr 0.0001, hidden 128
   - Adam lr 0.001, hidden 256
3. Berdasarkan **ROUGE-L**, konfigurasi terbaik adalah **RMSprop lr 0.001, hidden 128** dengan ROUGE-1 = 0,0707 dan ROUGE-L = 0,0696.
4. Berdasarkan **validation loss**, konfigurasi terbaik adalah **Adam lr 0.001, hidden 128** dengan best validation loss = 6,4938.
5. Hidden dimension 256 meningkatkan waktu training sekitar 1,52× dan jumlah parameter, tetapi tidak meningkatkan ROUGE secara konsisten.
6. Learning rate 0.0001 terlalu kecil untuk 8 epoch; model underfit dan menghasilkan ringkasan kosong.
7. Kualitas ringkasan masih rendah. Penyebab utamanya adalah keterbatasan Vanilla RNN (bottleneck satu hidden state dan vanishing gradient), mode collapse ke frasa umum, serta data/epoch yang terbatas.

### 5.3 Temuan Lintas Eksperimen

1. **Vanilla RNN masih berguna untuk klasifikasi sequence pendek, tetapi belum memadai untuk summarization.** Klasifikasi many-to-one mencapai 77,50%, sedangkan Seq2Seq many-to-many hanya menghasilkan ROUGE di bawah 0,08 dan mengalami mode collapse.
2. **Pemilihan arsitektur lebih penting daripada tuning hyperparameter tunggal.** Pada klasifikasi, GRU/LSTM meningkatkan validation F1 dari 0,6237 menjadi sekitar 0,861 dan menurunkan simpangan baku antar seed dari 0,192 menjadi 0,024. Perubahan optimizer atau hyperparameter tunggal tidak memberikan peningkatan yang konsisten.
3. **Data kecil membuat evaluasi tidak pasti.** Dengan 40 data uji, satu kesalahan setara 2,5 poin persentase; interval kepercayaan 95% melebar dan saling tumpang tindih. Dengan 24 data validasi, satu prediksi berubah setara 4,17 poin persentase.
4. **Validation loss bukan pengganti metrik task-level.** Summarization menunjukkan validation loss terbaik tidak menghasilkan ROUGE terbaik; klasifikasi menunjukkan validation loss terbaik tidak selalu berarti accuracy validasi tertinggi.
5. **Mode collapse merupakan kegagalan yang lebih serius daripada overfitting biasa.** Model summarization menghasilkan satu prediksi identik untuk seluruh data uji, sehingga tidak belajar pemetaan kondisional artikel→ringkasan. Ini menunjukkan perlunya attention, arsitektur bergating, dan strategi decoding yang lebih baik.
6. **Kedua pipeline sudah reproducible dan siap dikembangkan.** Pemisahan modul, penyimpanan konfigurasi/vocabulary/bobot, penggunaan seed, serta artefak evaluasi memudahkan replikasi dan pengembangan lanjutan.

### 5.4 Saran Pengembangan

**Klasifikasi sentimen**

1. Menerapkan **k-fold cross-validation** dan beberapa random seed agar estimasi performa lebih stabil, bukan hanya dari satu split 133/24/40.
2. Menambahkan **normalisasi kata gaul**, **stemming Sastrawi**, dan **penanganan negasi** untuk memperkecil variasi kosakata.
3. Menggunakan **pretrained word embedding bahasa Indonesia** (misalnya fastText atau Word2Vec) agar kata yang jarang muncul tetap memiliki representasi yang bermakna.
4. Menambah jumlah data atau melakukan augmentasi data untuk mengurangi overfitting.
5. Melakukan **tuning yang lebih luas** (grid search, random search, atau Bayesian optimization) dengan validasi silang, karena tuning satu faktor pada satu waktu belum cukup untuk menemukan kombinasi optimal.
6. Menganalisis kesalahan per tweet untuk memahami pola sarkasme dan sentimen campuran, lalu menyesuaikan preprocessing atau model.
7. Melanjutkan perbandingan arsitektur dengan protokol yang lebih kuat. Hasil tuning sudah menunjukkan GRU/LSTM lebih baik pada validation set; langkah berikutnya adalah menguji kandidat tersebut dengan k-fold cross-validation atau nested cross-validation dan data yang lebih banyak.

**Text summarization**

1. Menambah jumlah data dan epoch. Dataset penuh (365.121 pasangan) dan durasi training yang lebih panjang berpotensi memperbaiki hasil, meskipun tetap dibatasi oleh kapasitas Vanilla RNN.
2. Menambahkan **attention mechanism** agar decoder dapat mengakses bagian artikel yang relevan dan tidak bergantung pada satu vektor hidden state.
3. Menggunakan **beam search** dan **scheduled sampling** untuk mengurangi exposure bias dan memperbaiki kualitas decoding.
4. Mencoba **subword tokenization** (misalnya BPE atau SentencePiece) untuk menangani kata langka dan memperkecil vocabulary.
5. Menambahkan **coverage mechanism** agar ringkasan tidak berulang dan mencakup informasi utama.
6. Sebagai pembanding di luar batasan tugas, menggunakan **LSTM/GRU atau Transformer** dengan protokol evaluasi yang sama.

**Lintas eksperimen**

1. Menstandarkan pelaporan hasil dengan interval kepercayaan dan beberapa split data agar perbandingan antar model lebih dapat dipercaya.
2. Mencatat waktu training, jumlah parameter, dan metrik task-level secara konsisten pada semua eksperimen.
3. Menyimpan analisis kesalahan kualitatif (contoh prediksi salah) untuk memahami pola kegagalan model, tidak hanya angka agregat.

---

## Lampiran A — Tabel Per-Epoch Summarization

### A.1 Training Loss

| Epoch | adam_lr0.0001_h128 | adam_lr0.001_h128 | adam_lr0.001_h256 | rmsprop_lr0.001_h128 |
|---:|---:|---:|---:|---:|
| 1 | 9,1814 | 7,4665 | 7,2001 | 7,4139 |
| 2 | 8,0558 | 6,4335 | 6,2971 | 6,5886 |
| 3 | 7,2498 | 6,2066 | 6,0671 | 6,4620 |
| 4 | 6,8725 | 6,0255 | 5,8891 | 6,3946 |
| 5 | 6,6835 | 5,8762 | 5,7181 | 6,3540 |
| 6 | 6,5919 | 5,7448 | 5,5462 | 6,3249 |
| 7 | 6,5437 | 5,6194 | 5,3923 | 6,2952 |
| 8 | 6,5090 | 5,5090 | 5,2133 | 6,2670 |

### A.2 Validation Loss

| Epoch | adam_lr0.0001_h128 | adam_lr0.001_h128 | adam_lr0.001_h256 | rmsprop_lr0.001_h128 |
|---:|---:|---:|---:|---:|
| 1 | 8,7193 | 6,8699 | 6,7410 | 6,7945 |
| 2 | 7,7253 | 6,7193 | 6,6343 | 6,6837 |
| 3 | 7,2694 | 6,5965 | 6,6019 | 6,6269 |
| 4 | 7,0502 | 6,5455 | 6,5617 | 6,6073 |
| 5 | 6,9524 | 6,5275 | 6,5650 | 6,5921 |
| 6 | 6,9136 | 6,5029 | 6,5435 | 6,5729 |
| 7 | 6,8987 | 6,4938 | 6,5920 | 6,5530 |
| 8 | 6,8797 | 6,5027 | 6,6066 | 6,5372 |

### A.3 Execution Time per Epoch (detik)

| Epoch | adam_lr0.0001_h128 | adam_lr0.001_h128 | adam_lr0.001_h256 | rmsprop_lr0.001_h128 |
|---:|---:|---:|---:|---:|
| 1 | 15,0658 | 14,5327 | 22,2934 | 14,1621 |
| 2 | 13,6733 | 12,6179 | 19,4097 | 12,2169 |
| 3 | 13,1342 | 12,8066 | 19,4048 | 12,2724 |
| 4 | 12,9807 | 12,7285 | 19,3870 | 12,3645 |
| 5 | 12,9873 | 12,7952 | 19,2721 | 12,2586 |
| 6 | 13,0056 | 12,8494 | 19,4796 | 12,3383 |
| 7 | 13,1164 | 12,8461 | 19,6971 | 12,2830 |
| 8 | 13,0802 | 13,0099 | 19,4600 | 12,2227 |

---

## Lampiran B — Struktur Kode dan Artefak

### B.1 Klasifikasi Sentimen

| File | Fungsi |
|---|---|
| `klasifikasi/klasifikasi_sentimen_rnn.py` | Pipeline utama klasifikasi sentimen: preprocessing, split, vektorisasi, model, training, evaluasi, dan penyimpanan artefak |
| `klasifikasi/eksperimen_tuning.py` | Eksperimen pembanding dan tuning 23 konfigurasi × 3 seed |
| `klasifikasi/dataset_tweet_sentiment_opini_film.csv` | Dataset sentimen tweet opini film |
| `klasifikasi/output_rnn/` | Model utama, vocabulary, kurva, confusion matrix, dan hasil evaluasi |
| `klasifikasi/output_rnn/tuning/` | Hasil tuning, ringkasan statistik, grafik, dan model terbaik |

**Artefak output klasifikasi**

| File | Keterangan |
|---|---|
| `klasifikasi/output_rnn/kurva_loss_accuracy.png` | Kurva loss dan accuracy training model utama |
| `klasifikasi/output_rnn/confusion_matrix.png` | Visualisasi confusion matrix data uji model utama |
| `klasifikasi/output_rnn/hasil_evaluasi.txt` | Ringkasan metrik evaluasi model utama |
| `klasifikasi/output_rnn/model_rnn_terbaik.keras` | Model utama dengan val_loss terbaik |
| `klasifikasi/output_rnn/vocabulary.txt` | Kosakata hasil TextVectorization |
| `klasifikasi/output_rnn/tuning/hasil_tuning_per_seed.csv` | Hasil 69 run training per konfigurasi dan seed |
| `klasifikasi/output_rnn/tuning/hasil_tuning_per_grup.csv` | Hasil run per kelompok eksperimen |
| `klasifikasi/output_rnn/tuning/ringkasan_tuning.csv` | Ringkasan 23 konfigurasi berdasarkan validation F1 |
| `klasifikasi/output_rnn/tuning/perbandingan_optimizer.png` | Grafik perbandingan optimizer |
| `klasifikasi/output_rnn/tuning/perbandingan_arsitektur.png` | Grafik perbandingan arsitektur RNN |
| `klasifikasi/output_rnn/tuning/tuning_hyperparameter.png` | Panel pengaruh hyperparameter |
| `klasifikasi/output_rnn/tuning/top_konfigurasi.png` | Grafik 10 konfigurasi terbaik |
| `klasifikasi/output_rnn/tuning/model_terbaik/model_tuning_terbaik.keras` | Model dari konfigurasi tuning terbaik |
| `klasifikasi/output_rnn/tuning/model_terbaik/kurva_model_tuning_terbaik.png` | Kurva training konfigurasi terbaik |
| `klasifikasi/output_rnn/tuning/model_terbaik/confusion_matrix.png` | Confusion matrix konfigurasi terbaik pada data uji |
| `klasifikasi/output_rnn/tuning/model_terbaik/hasil_evaluasi.txt` | Classification report konfigurasi terbaik |
| `klasifikasi/output_rnn/tuning/model_terbaik/perbandingan_baseline_vs_terbaik.csv` | Perbandingan metrik test baseline vs GRU terbaik |
| `klasifikasi/output_rnn/tuning/model_terbaik/konfigurasi_terbaik.json` | Detail konfigurasi dan metrik terbaik |

### B.2 Text Summarization

| File | Fungsi |
|---|---|
| `summarize/config.py` | Konfigurasi data, hyperparameter, dan daftar eksperimen |
| `summarize/preprocessing.py` | Pembersihan teks, tokenisasi, Vocabulary, padding |
| `summarize/data.py` | Pembacaan `wikihow.json`, split per halaman, encoding ke tensor |
| `summarize/model.py` | Encoder, Decoder, Seq2Seq, masked loss, optimizer |
| `summarize/rouge.py` | Implementasi ROUGE-1, ROUGE-2, ROUGE-L |
| `summarize/training.py` | Training, timer per epoch, evaluasi, tabel, grafik |
| `summarize/experiments.py` | CLI menjalankan seluruh konfigurasi |
| `summarize/inference.py` | Greedy autoregressive decoding |
| `summarize/summarize.py` | Fungsi `summarize(text)` dan CLI inference |
| `summarize/evaluate.py` | Menampilkan tabel evaluasi dari `result.json` |
| `summarize/report.md` | Laporan asli eksperimen summarization |
| `summarize/requirements.txt` | Dependensi Python |

**Artefak hasil eksperimen summarization**

```text
summarize/outputs/
├── best_model/                  # model terbaik + vocab + konfigurasi
├── experiments/
│   ├── adam_lr0.001_h128/
│   ├── rmsprop_lr0.001_h128/
│   ├── adam_lr0.0001_h128/
│   └── adam_lr0.001_h256/
├── comparison.csv
├── comparison.md
├── evaluation_table.csv
├── loss_curves.png
├── epoch_times.png
├── rouge_comparison.png
├── per_epoch_train_loss.md
├── per_epoch_val_loss.md
└── per_epoch_epoch_time_sec.md
```

Setiap direktori konfigurasi berisi:

- `model.weights.h5`
- `vocab.json`
- `model_config.json`
- `history.csv`
- `test_predictions.csv`
- `result.json`

### B.3 Dokumen Sumber

| File | Keterangan |
|---|---|
| `klasifikasi/report.md` | Laporan asli eksperimen klasifikasi sentimen |
| `summarize/report.md` | Laporan asli eksperimen text summarization |
| `report.md` | Laporan gabungan (dokumen ini) |

---

## Lampiran C — Lingkungan Eksperimen

| Item | Nilai |
|---|---|
| OS | Windows |
| Python | 3.13.1 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.0 |
| NumPy | 2.2.2 |
| Matplotlib | 3.10.3 |
| Seaborn | 0.13.2 |
| GPU | Tidak tersedia (CPU-only) |
| Seed | 42 |

Keterbatasan perangkat keras menjadi salah satu alasan utama pembatasan jumlah data dan epoch pada eksperimen summarization. Pada eksperimen klasifikasi, dataset yang kecil membuat training relatif cepat, tetapi juga membatasi kemampuan generalisasi model.

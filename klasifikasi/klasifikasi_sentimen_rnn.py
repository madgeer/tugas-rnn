"""
Klasifikasi Sentimen Tweet Opini Film dengan Recurrent Neural Network (RNN)
===========================================================================

Task      : Klasifikasi teks many-to-one (satu dokumen/tweet -> satu label)
Dataset   : dataset_tweet_sentiment_opini_film.csv
Library   : TensorFlow / Keras, scikit-learn, pandas, matplotlib, seaborn

Instalasi dependensi (jika belum tersedia):
    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

Cara menjalankan:
    python klasifikasi_sentimen_rnn.py

Output:
    - Kurva training (loss & accuracy)  -> output_rnn/kurva_loss_accuracy.png
    - Confusion matrix                  -> output_rnn/confusion_matrix.png
    - Ringkasan evaluasi                -> output_rnn/hasil_evaluasi.txt
    - Model terbaik                     -> output_rnn/model_rnn_terbaik.keras
    - Kosakata hasil vektorisasi        -> output_rnn/vocabulary.txt
"""

from __future__ import annotations

import html
import os
import random
import re
from pathlib import Path

# Kurangi log verbosity TensorFlow. Harus di-set sebelum TensorFlow diimpor.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ============================== KONFIGURASI ==============================
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_tweet_sentiment_opini_film.csv"
OUTPUT_DIR = BASE_DIR / "output_rnn"

TEST_SIZE = 0.20       # 20% data uji
VAL_SIZE = 0.15        # 15% dari data latih dipakai sebagai validation set
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 5e-4

MAX_TOKENS = 10_000    # ukuran maksimum kosakata
EMBEDDING_DIM = 64
EMBEDDING_L2 = 1e-4    # regularisasi L2 pada embedding (mencegah overfitting)
RNN_UNITS = 32
RNN_DROPOUT = 0.5      # dropout pada input RNN
RECURRENT_DROPOUT = 0.2  # dropout pada koneksi rekuren
DROPOUT = 0.5          # dropout pada layer dense

# Pilihan layer rekuren: "simple_rnn", "gru", atau "lstm".
# Default SimpleRNN agar sesuai dengan permintaan "model RNN".
# Ganti ke "lstm"/"gru" jika ingin membandingkan varian RNN yang lebih kuat.
RNN_TYPE = "simple_rnn"

MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 60
PADDING_PERCENTILE = 95
# ========================================================================


def set_seed(seed: int) -> None:
    """Menyetel seed agar hasil training dapat direproduksi."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    """Membaca dataset lokal dan menyeragamkan nama kolom."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file dataset_tweet_sentiment_opini_film.csv berada "
            "di folder yang sama dengan skrip ini."
        )

    df = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("Dataset tidak dapat dibaca. Periksa format/encoding file CSV.")

    # Deteksi nama kolom teks dan label (mendukung beberapa variasi nama).
    text_aliases = {"text tweet", "text", "tweet", "ulasan", "komentar", "review"}
    label_aliases = {"sentiment", "sentimen", "label", "class", "kelas"}

    rename_map = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key in text_aliases:
            rename_map[column] = "text"
        elif key in label_aliases:
            rename_map[column] = "label"

    df = df.rename(columns=rename_map)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            "Kolom 'text' dan 'label' tidak ditemukan. "
            f"Kolom yang tersedia: {df.columns.tolist()}"
        )

    return df[["text", "label"]].copy()


def clean_text(text: object) -> str:
    """
    Preprocessing teks dasar:
    1. Cleaning     : hapus URL, mention, RT, dan HTML entity.
    2. Case folding : ubah seluruh teks menjadi huruf kecil.
    3. Hapus karakter khusus: tanda baca, angka, emoji, simbol, dsb.
    4. Normalisasi spasi berlebih.
    """
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    # Cleaning: URL, mention, RT, dan tagar (kata di dalam tagar dipertahankan).
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Case folding.
    text = text.lower()

    # Hapus semua karakter selain huruf a-z dan spasi
    # (tanda baca, angka, emoji, simbol, dan karakter khusus ikut terhapus).
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalisasi spasi.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan teks, label, serta membuang data kosong/duplikat."""
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text_clean"] = df["text"].map(clean_text)

    # Buang teks kosong dan duplikat berdasarkan teks yang sudah dibersihkan.
    before = len(df)
    df = df[df["text_clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    removed = before - len(df)

    if len(df) == 0:
        raise ValueError("Semua data kosong setelah preprocessing.")

    print(f"Jumlah data awal          : {before}")
    print(f"Data kosong/duplikat dibuang: {removed}")
    print(f"Jumlah data siap pakai    : {len(df)}")
    return df


def determine_sequence_length(texts: np.ndarray) -> int:
    """Menentukan panjang sequence berdasarkan persentil jumlah kata data latih."""
    lengths = np.array([len(text.split()) for text in texts], dtype=int)
    percentile_value = int(np.ceil(np.percentile(lengths, PADDING_PERCENTILE)))
    sequence_length = int(
        np.clip(percentile_value, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
    )
    print(
        f"Panjang sequence (persentil {PADDING_PERCENTILE} jumlah kata): "
        f"{sequence_length} token"
    )
    return sequence_length


def build_vectorizer(
    train_texts: np.ndarray, max_length: int
) -> tf.keras.layers.TextVectorization:
    """Membuat dan meng-adapt TextVectorization HANYA pada data latih."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        standardize=None,  # teks sudah dibersihkan pada tahap preprocessing
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_length,
        name="text_vectorization",
    )
    vectorizer.adapt(train_texts)
    return vectorizer


def vectorize_texts(
    vectorizer: tf.keras.layers.TextVectorization,
    texts: np.ndarray,
) -> np.ndarray:
    """Mengubah teks menjadi sequence integer dengan vectorizer yang sudah di-adapt."""
    sequences = vectorizer(texts)
    return np.asarray(sequences, dtype=np.int32)


def build_model(
    vocab_size: int, max_length: int, rnn_type: str
) -> tf.keras.Model:
    """
    Arsitektur many-to-one:
    sequence token -> embedding -> RNN (hanya output terakhir) ->
    dense -> sigmoid (probabilitas kelas positif).
    """
    rnn_layers = {
        "simple_rnn": tf.keras.layers.SimpleRNN,
        "gru": tf.keras.layers.GRU,
        "lstm": tf.keras.layers.LSTM,
    }
    rnn_type = rnn_type.lower()
    if rnn_type not in rnn_layers:
        raise ValueError(
            f"RNN_TYPE='{rnn_type}' tidak dikenal. "
            f"Gunakan salah satu dari: {list(rnn_layers)}"
        )

    inputs = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="token_ids")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        embeddings_regularizer=tf.keras.regularizers.l2(EMBEDDING_L2),
        name="embedding",
    )(inputs)

    # return_sequences=False (default) => many-to-one:
    # seluruh urutan diringkas menjadi satu vektor keadaan terakhir.
    x = rnn_layers[rnn_type](
        units=RNN_UNITS,
        dropout=RNN_DROPOUT,
        recurrent_dropout=RECURRENT_DROPOUT,
        name=f"{rnn_type}_many_to_one",
    )(x)

    x = tf.keras.layers.Dense(32, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sentimen")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rnn_sentimen")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history: tf.keras.callbacks.History, save_path: Path) -> None:
    """Menampilkan dan menyimpan kurva loss serta accuracy."""
    sns.set_theme(style="whitegrid")
    history_dict = history.history
    epoch_axis = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(epoch_axis, history_dict["loss"], "o-", label="Training Loss")
    axes[0].plot(
        epoch_axis, history_dict["val_loss"], "s-", label="Validation Loss"
    )
    axes[0].set_title("Kurva Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()

    axes[1].plot(epoch_axis, history_dict["accuracy"], "o-", label="Training Accuracy")
    axes[1].plot(
        epoch_axis, history_dict["val_accuracy"], "s-", label="Validation Accuracy"
    )
    axes[1].set_title("Kurva Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1
    for axis in axes:
        axis.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Epoch terbaik ({best_epoch})",
        )
        axis.legend()

    fig.suptitle("Kurva Training Model RNN - Klasifikasi Sentimen", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Kurva training disimpan di: {save_path}")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, label_names: list[str], save_path: Path
) -> None:
    """Menampilkan dan menyimpan confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Confusion Matrix - Data Uji")
    plt.xlabel("Label Prediksi")
    plt.ylabel("Label Aktual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix disimpan di: {save_path}")
    plt.show()
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    save_dir: Path,
) -> dict[str, float]:
    """Evaluasi data uji: confusion matrix + classification report + metrik akhir."""
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual: {name}" for name in label_names],
        columns=[f"Prediksi: {name}" for name in label_names],
    )

    lines = []
    lines.append("=" * 70)
    lines.append("HASIL EVALUASI PADA DATA UJI")
    lines.append("=" * 70)
    lines.append(f"Jumlah data uji : {len(y_test)}")
    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append(cm_df.to_string())
    lines.append("")
    lines.append("Classification Report:")
    lines.append(report)
    lines.append("Metrik Akhir (weighted average):")
    lines.append(f"  Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    lines.append(f"  Precision : {precision:.4f}")
    lines.append(f"  Recall    : {recall:.4f}")
    lines.append(f"  F1-Score  : {f1:.4f}")
    lines.append("")
    lines.append("Metrik tambahan (macro average):")
    lines.append(
        f"  Precision : {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    lines.append(
        f"  Recall    : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    lines.append(
        f"  F1-Score  : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    lines.append("=" * 70)

    result_text = "\n".join(lines)
    print("\n" + result_text)

    plot_confusion_matrix(cm, label_names, save_dir / "confusion_matrix.png")

    report_path = save_dir / "hasil_evaluasi.txt"
    report_path.write_text(result_text + "\n", encoding="utf-8")
    print(f"Ringkasan evaluasi disimpan di: {report_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def show_sample_predictions(
    model: tf.keras.Model,
    vectorizer: tf.keras.layers.TextVectorization,
    label_names: list[str],
) -> None:
    """Contoh inferensi many-to-one pada kalimat baru (opsional)."""
    samples = [
        "Film ini sangat bagus, ceritanya menarik dan aktingnya memukau!",
        "Jelek banget filmnya, membosankan dan tidak worth it untuk ditonton.",
        "Biasa saja, tidak ada yang istimewa dari film ini.",
    ]
    cleaned = np.array([clean_text(text) for text in samples], dtype=object)
    sequences = vectorize_texts(vectorizer, cleaned)
    probabilities = model.predict(sequences, verbose=0).ravel()

    print("\nContoh Prediksi Teks Baru:")
    for original, probability in zip(samples, probabilities):
        predicted_index = int(probability >= 0.5)
        confidence = probability if predicted_index == 1 else 1 - probability
        print(
            f"  - {original}\n"
            f"    -> {label_names[predicted_index]} "
            f"(keyakinan: {confidence:.2%})"
        )


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KLASIFIKASI SENTIMEN TWEET OPINI FILM DENGAN RNN")
    print("=" * 70)

    # 1. Membaca dan preprocessing dataset.
    raw_df = load_dataset(DATASET_PATH)
    df = preprocess_dataset(raw_df)

    label_names = sorted(df["label"].unique().tolist())
    if len(label_names) != 2:
        raise ValueError(
            "Skrip ini dirancang untuk klasifikasi biner, tetapi ditemukan "
            f"{len(label_names)} kelas: {label_names}"
        )
    label_to_id = {name: index for index, name in enumerate(label_names)}
    df["label_id"] = df["label"].map(label_to_id)
    y = df["label_id"].to_numpy(dtype=np.int32)
    X = df["text_clean"].to_numpy(dtype=object)

    print(f"Kelas sentimen            : {label_names} {label_to_id}")
    print("Distribusi kelas:")
    print(df["label"].value_counts().to_string())

    print("\nContoh preprocessing teks:")
    for _, row in df.head(3).iterrows():
        print(f"  Sebelum : {row['text']}")
        print(f"  Sesudah : {row['text_clean']}")
        print()

    # 2. Train-test split (stratified agar proporsi kelas seimbang).
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    # Validation set diambil dari data latih, sehingga data uji tetap murni
    # dan hanya dipakai sekali untuk evaluasi akhir.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train_full,
    )

    print("\nPembagian data:")
    print(f"  Data latih      : {len(X_train)}")
    print(f"  Data validasi   : {len(X_val)}")
    print(f"  Data uji        : {len(X_test)}")

    # 3. Vektorisasi teks (fit hanya pada data latih untuk mencegah data leakage).
    max_length = determine_sequence_length(X_train)
    vectorizer = build_vectorizer(X_train, max_length)
    vocab_size = vectorizer.vocabulary_size()
    print(f"Ukuran kosakata hasil adapt: {vocab_size} token")

    X_train_seq = vectorize_texts(vectorizer, X_train)
    X_val_seq = vectorize_texts(vectorizer, X_val)
    X_test_seq = vectorize_texts(vectorizer, X_test)
    print(
        f"Bentuk data setelah vektorisasi -> train: {X_train_seq.shape}, "
        f"val: {X_val_seq.shape}, test: {X_test_seq.shape}"
    )

    # Simpan kosakata agar model dapat dipakai ulang untuk inferensi.
    vocab_path = OUTPUT_DIR / "vocabulary.txt"
    vocab_path.write_text(
        "\n".join(vectorizer.get_vocabulary()), encoding="utf-8"
    )
    print(f"Kosakata disimpan di: {vocab_path}")

    # 4. Membangun model RNN many-to-one.
    model = build_model(vocab_size, max_length, RNN_TYPE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "model_rnn_terbaik.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 5. Training model.
    print("\nMulai training...")
    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # 6. Kurva loss dan accuracy.
    plot_training_history(history, OUTPUT_DIR / "kurva_loss_accuracy.png")

    # 7. Evaluasi akhir pada data uji.
    evaluate_model(model, X_test_seq, y_test, label_names, OUTPUT_DIR)

    # 8. Contoh prediksi kalimat baru.
    show_sample_predictions(model, vectorizer, label_names)

    print("\nSelesai. Semua output tersimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

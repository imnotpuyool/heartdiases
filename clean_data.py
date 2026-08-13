import sys
sys.modules['numexpr'] = None
sys.modules['bottleneck'] = None

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_heart_data(file_path=None, output_path=None):
    if file_path is None:
        file_path = os.path.join(BASE_DIR, "heart.csv")
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "heart_cleaned.csv")

    print("=" * 50)
    print("PROSES EXPLORATION & DATA CLEANING")
    print("=" * 50)

    df = pd.read_csv(file_path)

    print(f"--> Dataset awal: {df.shape[0]} baris, {df.shape[1]} kolom.")

    null_counts = df.isnull().sum()
    print("\n1. Pengecekan Missing Values:")
    if null_counts.sum() == 0:
        print("   [OK] Tidak ada missing values.")
    else:
        print("   [WARNING] Missing values ditemukan:")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print("   -> Baris dengan missing value telah dihapus.")

    dup_count = df.duplicated().sum()
    print(f"\n2. Pengecekan Duplikat: Ditemukan {dup_count} baris duplikat.")
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"   [OK] Duplikat dihapus. Sisa baris: {len(df)}")

    print("\n3. Pengecekan Outlier/Nilai Anomali:")
    if 'thal' in df.columns:
        invalid_thal = (df['thal'] == 0).sum()
        if invalid_thal > 0:
            print(f"   [WARNING] {invalid_thal} baris thal=0 difilter.")
            df = df[df['thal'] != 0]

    if 'ca' in df.columns:
        invalid_ca = (df['ca'] == 4).sum()
        if invalid_ca > 0:
            print(f"   [WARNING] {invalid_ca} baris ca=4 difilter.")
            df = df[df['ca'] != 4]

    print("\n4. Distribusi Target Label:")
    print(df['target'].value_counts())

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Data bersih disimpan ke '{output_path}' ({len(df)} baris).")
    print("=" * 50)
    return output_path


if __name__ == "__main__":
    clean_heart_data()

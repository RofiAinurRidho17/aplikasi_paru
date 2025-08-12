import os
import pandas as pd

csv_path = 'data/user_predictions.csv'

columns = [
    'usia_muda',
    'usia_tua',
    'jenis_kelamin_pria',
    'jenis_kelamin_wanita',
    'merokok_aktif',
    'merokok_pasif',
    'bekerja_tidak',
    'bekerja_ya',
    'rumah_tangga_tidak',
    'rumah_tangga_ya',
    'aktivitas_begadang_tidak',
    'aktivitas_begadang_ya',
    'aktivitas_olahraga_jarang',
    'aktivitas_olahraga_sering',
    'asuransi_ada',
    'asuransi_tidak',
    'penyakit_bawaan_ada',
    'penyakit_bawaan_tidak',
    'prediction',
    'probability',
    'timestamp'
]

if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"File {csv_path} berhasil dihapus.")
else:
    print(f"File {csv_path} tidak ditemukan, membuat file baru.")

df_kosong = pd.DataFrame(columns=columns)
df_kosong.to_csv(csv_path, index=False)
print(f"File baru {csv_path} berhasil dibuat dengan header.")

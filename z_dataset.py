import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

data = {
    'user_id': [f'USR{i:03d}' for i in range(1, n + 1)],
    'age': np.random.randint(18, 65, size=n),
    'tenure_months': np.random.randint(1, 48, size=n),
    'monthly_spend': np.round(np.random.uniform(20.0, 300.0, size=n), 2),
    'gender': np.random.choice(['Male', 'Female', np.nan], size=n, p=[0.45, 0.45, 0.10]),
    'subscription_type': np.random.choice(['Basic', 'Premium', 'VIP'], size=n),
    'payment_method': np.random.choice(['Credit Card', 'E-Wallet', 'Bank Transfer'], size=n),
    'churn': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
}

df_dummy = pd.DataFrame(data)

# Buat beberapa missing value di monthly_spend
df_dummy.loc[df_dummy.sample(5).index, 'monthly_spend'] = np.nan

# Tambahkan 3 baris duplikat untuk dites oleh quick_eda
df_dummy = pd.concat([df_dummy, df_dummy.iloc[:3]], ignore_index=True)

# Simpan ke CSV
df_dummy.to_csv('dataset_peserta.csv', index=False)
print("Berhasil membuat dataset_peserta.csv (103 baris data)")
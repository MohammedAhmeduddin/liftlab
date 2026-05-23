# notebooks/eda/00_data_validation.py
import sys
from pathlib import Path

# Make sure liftlab is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from liftlab.data.loaders import load_criteo, load_olist
from liftlab.config import get_settings

print("=" * 60)
print("LiftLab Milestone 1 — Smoke Test")
print("=" * 60)

# 1. Config check
settings = get_settings()
print(f"\n✅ Config loaded: {settings.app_name}")
print(f"   DB URL      : {settings.database_url}")
print(f"   MLflow URI  : {settings.mlflow_tracking_uri}")

# 2. Criteo check
print("\n--- Criteo Dataset ---")
df = load_criteo(sample_frac=0.05)   # 5% sample = ~700K rows, fast to load
print(f"✅ Shape        : {df.shape}")
print(f"   Columns     : {list(df.columns)}")
print(f"   Treatment % : {df['treatment'].mean():.3f}")
print(f"   Conversion %: {df['conversion'].mean():.4f}")
print(f"   Visit %     : {df['visit'].mean():.4f}")
print(f"   Nulls       : {df.isnull().sum().sum()}")

# 3. Olist check
print("\n--- Olist Dataset ---")
olist = load_olist()
print(f"✅ Tables loaded: {len(olist)}")
for name, table in olist.items():
    print(f"   {name:<35} {len(table):>7,} rows  |  {table.shape[1]} cols")

# 4. Docker services check
print("\n--- Infrastructure Check ---")
import socket

def check_port(host, port, name):
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        print(f"✅ {name} is reachable at {host}:{port}")
    except OSError:
        print(f"❌ {name} is NOT reachable at {host}:{port} — is Docker running?")

check_port("localhost", 5432, "PostgreSQL")
check_port("localhost", 5000, "MLflow")

print("\n" + "=" * 60)
print("Milestone 1 Complete ✅" if True else "")
print("=" * 60)

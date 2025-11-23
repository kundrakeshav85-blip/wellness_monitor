import os, h5py
p = r"Emotion-detection\src\model.h5"
print("exists:", os.path.exists(p))
print("size bytes:", os.path.getsize(p) if os.path.exists(p) else "n/a")
try:
    f = h5py.File(p, "r")
    print("h5 keys:", list(f.keys()))
    f.close()
except Exception as e:
    print("h5py error:", e)

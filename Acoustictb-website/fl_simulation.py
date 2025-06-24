import joblib, numpy as np, xgboost as xgb

# 1) Load the saved model, scaler, and your local data
model = joblib.load("tb_detection_model.joblib")
scaler = joblib.load("scaler.joblib")
X_local = np.load("X_processed.npy")
y_local = np.load("y_processed.npy")

#manually run 3 rounds of “federated” training with your one client
for rnd in range(1, 4):
    print(f"--- Round {rnd} ---")
    #extract the current “global” parameters
    raw = model.get_booster().save_raw()

    #deserialize into your client
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    model._Booster = booster

    #fit locally
    print(f"Training locally on {len(y_local)} samples")
    model.fit(X_local, y_local, xgb_model=model.get_booster())
    print("Local update applied.\n")

joblib.dump(model, "global_after_fl_demo.joblib")
print("Demo federated learning complete.")

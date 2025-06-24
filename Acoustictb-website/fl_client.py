import flwr as fl
from flwr.client import NumPyClient
import joblib
import numpy as np
import xgboost as xgb
model = joblib.load("tb_detection_model.joblib")

class XGBClient(NumPyClient):
    def get_parameters(self):
        raw = model.get_booster().save_raw()
        return [memoryview(raw).tolist()]

    def fit(self, parameters, config):
        #deserialize global parameters
        booster = xgb.Booster()
        booster.load_model(bytearray(parameters[0]))
        model._Booster = booster
        X_local = np.load("X_processed.npy")
        y_local = np.load("y_processed.npy")
        #local training
        model.fit(X_local, y_local, xgb_model=model.get_booster())
        raw_updated = model.get_booster().save_raw()
        return [memoryview(raw_updated).tolist()], len(y_local), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",  #must match the server
        client=XGBClient(),
    )

import flwr as fl
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
import joblib

model = joblib.load("tb_detection_model.joblib")

def get_initial_parameters():
    raw = model.get_booster().save_raw()
    #flower expects a list of raw‐byte buffers
    return [memoryview(raw).tolist()]

#only one client in this demo
strategy = FedAvg(
    fraction_fit=1.0,          
    fraction_evaluate=0.0,   
    min_fit_clients=1,        
    min_available_clients=1,   
    initial_parameters=get_initial_parameters(),
)

if __name__ == "__main__":
    start_server(
        server_address="127.0.0.1:8080",
        config=ServerConfig(num_rounds=3, round_timeout=300),
        strategy=strategy,
    )

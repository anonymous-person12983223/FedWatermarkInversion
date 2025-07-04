from src.training.client.client import Client
from src.training.server.server import CustomServer
import flwr as fl
import os

class FederatedExperiment:
    def __init__(self, args, model_cls, dataset):
        self.args = args
        self.model_cls = model_cls
        self.dataset = dataset

    def create_client(self, cid, dataset):
        return Client(self.args, self.model_cls, cid, dataset)

    def run(self):
        # Total resources for simulation
        num_cpu = self.args.num_cpu
        num_gpu = self.args.num_gpu

        client_ids = ["c{}".format(i) for i in range(0, self.args.clients)]
        dataset = self.dataset
        dataset.init_train_split(client_ids)

        if not self.args.pretrain:
            model = self.model_cls()
        else:
            model = self.model_cls(self.args.pretrain)

        server = CustomServer(self.args, model, dataset)
        fl.simulation.start_simulation(client_fn=lambda x: self.create_client(x, dataset),
                                       num_clients=len(client_ids),
                                       clients_ids=client_ids,
                                       server=server,
                                       config=fl.server.ServerConfig(num_rounds=self.args.num_rounds),
                                       client_resources={"num_cpus": self.args.client_cpu,
                                                         "num_gpus": self.args.client_cuda},
                                       ray_init_args={"num_cpus": num_cpu,
                                                      "num_gpus": num_gpu,
                                                      "include_dashboard": True
                                                      #"log_to_driver": True,
                                                      #"logging_level": "DEBUG"})
                                                      })
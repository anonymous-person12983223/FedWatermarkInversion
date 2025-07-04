import flwr as fl
from src.utils.model_utils import train_model, evaluate_model
import torch
from collections import OrderedDict

class Client(fl.client.NumPyClient):
    """
    Federated learning client implementation using Flower's NumPyClient interface.

    This client handles the local training and evaluation of a model using PyTorch,
    in a federated learning setup.

    Attributes:
        args (Namespace): Configuration arguments (e.g., learning rate, device, etc.).
        model_fn (Callable): Function that returns a fresh instance of the model.
        cid (str): Client ID assigned by the server.
        data: Data loader object that provides training data.
        model (torch.nn.Module): The model to be trained locally.
        train_samples_idx (List[int]): Indexes of training samples for this client.
    """

    def __init__(self, args, model_fn, cid, data):
        self.args = args
        self.train_samples_idx = []
        self.model = model_fn()
        self.cid = cid
        self.data = data

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model on local data.

        Args:
            parameters (List[np.ndarray]): Model parameters from the server.
            config (dict): Configuration dictionary from the server.

        Returns:
            Tuple: A tuple containing:
                - List[np.ndarray]: Updated model parameters.
                - int: Number of training samples used.
                - dict: Metrics before and after training (e.g., accuracy).
        """
        # Load global model parameters into the local model
        self.set_parameters(parameters)

        # Load training data specific to this client
        dataloader = self.data.get_trainloader(self)

        # Evaluate the model before training (on local data)
        metrics_before = evaluate_model(self.model, dataloader, device=self.args.device)

        # Train the model on local data
        train_model(
            self.model,
            dataloader,
            self.args.epoch,
            self.args.lr,
            self.args.momentum,
            self.args.decay,
            device=self.args.device
        )

        # Evaluate the model after training (on local data)
        metrics_after = evaluate_model(self.model, dataloader, device=self.args.device)

        return (
            self.get_parameters(config={}),
            len(self.train_samples_idx),
            {
                "train_accuracy": metrics_before["accuracy"],
                "train_accuracy_after_training": metrics_after["accuracy"]
            }
        )

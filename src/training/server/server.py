import copy
import os.path
from collections import OrderedDict
from flwr.server import Server, SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import torch
from src.dataloader.watermark_data import WatermarkDataset
from src.utils.logger import Logger
from src.utils.model_utils import evaluate_model, watermark_model
import numpy as np

def mean(metrics):
    """
    Aggregate training accuracy metrics from multiple clients.

    Args:
        metrics (list): List of tuples where each tuple contains
                        (client_id, metrics_dict) reported from clients.

    Returns:
        dict: Aggregated metrics including mean training accuracy,
              mean training accuracy after training, and watermark
              accuracy info if available.
    """
    metrics_list = list(map(lambda x: x[1]["train_accuracy"], metrics))
    mean_accuracy = np.mean(metrics_list)

    metrics_list2 = list(map(lambda x: x[1]["train_accuracy_after_training"], metrics))
    mean_accuracy2 = np.mean(metrics_list2)

    output = {"mean_train_accuracy": mean_accuracy, "mean_train_accuracy_after_training": mean_accuracy2}

    for client_metrics in metrics:
        if "wm_accuracy_before_aggregation" in client_metrics[1]:
            output["wm_epochs"] = client_metrics[1]["wm_epochs"]
            output["wm_accuracy_before_aggregation"] = client_metrics[1]["wm_accuracy_before_aggregation"]

    return output


class CustomServer(Server):
    """
    Custom federated learning server extending Flower's Server class.

    Handles:
    - Initialization of model and data
    - Watermarking the model periodically
    - Saving model checkpoints
    - Aggregating metrics and logging with wandb
    - Providing evaluation function for federated rounds

    Args:
        args: Namespace or dict containing configurations and hyperparameters.
        model: PyTorch model to train and aggregate.
        data: Dataset wrapper object providing training and test data loaders.
    """

    def __init__(self, args, model, data):
        """
        Initialize the custom server with model, data, and federated strategy.

        Sets up the federated averaging strategy with custom evaluation
        and fit metric aggregation, and prepares watermarking if enabled.
        """
        self.args = args
        self.model = model
        self.data = data
        self.mean_train_accuracy_history = []
        self.watermark_accuracy_history = []
        self.watermark_accuracy_before_aggregation_history = []
        self.cur_wm_threshold = 0.0
        self.logger = Logger()

        if self.args.watermark:
            self.wm_data = WatermarkDataset(self.args, (data.image_shape[0], data.image_shape[1]),
                                            self.data.num_classes, grayscale=data.image_shape[2] == 1)
            if args.watermark:
                self.watermark(0)
            self.save_model(0)
            initial_parameters_array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            initial_parameters = ndarrays_to_parameters(initial_parameters_array)
        else:
            initial_parameters_array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            initial_parameters = ndarrays_to_parameters(initial_parameters_array)

        fraction = self.args.clients_per_round / self.args.clients
        strategy = FedAvg(evaluate_fn=self.get_evaluate_fn(), fraction_evaluate=0.0, fraction_fit=fraction,
                          min_fit_clients=self.args.clients_per_round,
                          min_available_clients=self.args.clients,
                          initial_parameters=initial_parameters,
                          fit_metrics_aggregation_fn=mean)

        cm = SimpleClientManager()
        super().__init__(client_manager=cm, strategy=strategy)

    def watermark(self, server_round, epochs=None, threshold=None):
        """
        Apply watermark training to the global model to embed or refresh the watermark.

        Args:
            server_round (int): Current federated learning round.
            epochs (int, optional): Number of epochs to train on watermark data.
            threshold (float, optional): Accuracy threshold for watermark training.

        Side Effects:
            Updates the model in-place by training on watermark data.
            Logs watermark metrics to wandb.
            Records watermark accuracy history.
        """
        self.wm_data = WatermarkDataset(self.args,
                                        (self.data.image_shape[0], self.data.image_shape[1]),
                                        self.data.num_classes, grayscale=self.data.image_shape[2] == 1)
        wm_loader = self.wm_data.get_loader()

        # watermark accuracy before
        wm_result_dict_before = evaluate_model(self.model, wm_loader, self.args.device)
        print(wm_result_dict_before)

        # train model on wm
        if server_round == 0:
            threshold = threshold or 1.0
            lr = self.args.lr_pretrain
            momentum = self.args.momentum_pretrain
            decay = self.args.decay_pretrain
        else:
            threshold = threshold or self.args.watermark_threshold
            lr = self.args.lr_retrain
            momentum = self.args.momentum_retrain
            decay = self.args.decay_retrain
        start_acc = wm_result_dict_before["accuracy"]

        print("threshold", threshold)
        _, wm_epochs = watermark_model(self.model, wm_loader, epochs, threshold,
                                       lr, momentum, decay, self.args.device, start_acc)

        # watermark accuracy after
        wm_result_dict_after = evaluate_model(self.model, wm_loader, self.args.device)
        metrics_aggregated = {}
        metrics_aggregated.update({"before_wm_" + k: v for k, v in wm_result_dict_before.items()})
        metrics_aggregated.update({"after_wm_" + k: v for k, v in wm_result_dict_after.items()})
        metrics_aggregated["retrain_epochs"] = wm_epochs
        self.watermark_accuracy_history.append(wm_result_dict_after["accuracy"])
        self.logger.log(metrics_aggregated, step=server_round)

    def save_model(self, server_round):
        """
        Save the current global model state dict at specified rounds.

        Args:
            server_round (int): Current federated learning round.

        Saves the model to both the local output directory and wandb
        run directory for checkpointing and later analysis.
        """
        if self.args.model_checkpoint_rounds and server_round in self.args.model_checkpoint_rounds:
            filename = f"round_{server_round}.pth"
            dir_path = self.args.output_dir
            model_path = os.path.join(dir_path, filename)
            os.makedirs(dir_path, exist_ok=True)
            torch.save(self.model.state_dict(), str(model_path))

    def fit_round(
            self,
            server_round,
            timeout):
        """
        Perform a federated training round by aggregating client updates,
        updating the global model, and optionally applying watermarking.

        Args:
            server_round (int): Current federated learning round.
            timeout (float): Timeout duration for waiting client updates.

        Returns:
            Tuple: Aggregated parameters, aggregated metrics, and client results.
        """
        parameters_aggregated, metrics_aggregated, (results, failures) = super().fit_round(server_round, timeout)
        print(server_round)

        parameters_aggregated_array = parameters_to_ndarrays(parameters_aggregated)
        params_dict = zip(self.model.state_dict().keys(), parameters_aggregated_array)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.to("cpu")

        self.model.load_state_dict(state_dict, strict=True)
        print("assign model")

        self.mean_train_accuracy_history.append(metrics_aggregated["mean_train_accuracy"])
        print("train_accuracy", metrics_aggregated["mean_train_accuracy"])

        if self.args.watermark:
            self.watermark(server_round, self.args.max_watermark_epoch)

        wandb_metrics = copy.deepcopy(metrics_aggregated)
        self.save_model(server_round)
        # get weights
        parameters_aggregated_array = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated_array)
        self.logger.log(wandb_metrics, step=server_round)
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def get_evaluate_fn(self):
        """
        Create and return an evaluation function compatible with Flower's strategy.

        The evaluation function:
        - Loads given model parameters into the global model
        - Evaluates the global model on the test dataset
        - Logs evaluation metrics to wandb

        Returns:
            Callable: A function accepting (server_round, parameters, config) that
                      returns loss and metrics dictionary for evaluation.
        """

        def evaluate(server_round, parameters, config):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            test_set = self.data.get_testloader()
            result_dict = evaluate_model(self.model, test_set, self.args.device)
            self.logger.log(result_dict, step=server_round)
            return result_dict["loss"], result_dict

        return evaluate


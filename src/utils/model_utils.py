import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report


def train_model(model, dataloader, epochs=1, lr=0.01, momentum=0.2, weight_decay=0.0, device="cpu", report_fn=None):
    """
    Train a PyTorch model using SGD optimizer and cross-entropy loss.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader providing training data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        momentum (float): Momentum for SGD.
        weight_decay (float): Weight decay (L2 penalty).
        device (str): Device to run training on ('cpu' or 'cuda').
        report_fn (Callable, optional): Optional function to report loss each epoch.

    Returns:
        torch.nn.Module: Trained model.
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        cnt = 0
        total_loss = 0
        for data, target in dataloader:
            cnt += 1
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if report_fn:
            report_fn(total_loss / cnt)
    return model


def evaluate_model(model, test_set, device="cpu", per_class_metrics=False):
    """
    Evaluate a trained model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_set (DataLoader): DataLoader providing test data.
        device (str): Device to run evaluation on ('cpu' or 'cuda').
        per_class_metrics (bool): Whether to include per-class precision/recall.

    Returns:
        dict: Dictionary containing loss, accuracy, F1 score,
              and optionally precision and recall per class.
    """
    criterion = torch.nn.CrossEntropyLoss()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_set:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    average_loss = total_loss / total_samples
    correct_predictions = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_predictions, average='macro')

    if per_class_metrics:
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        precision_per_class = {
            f'label{i}_precision': class_report[str(i)]['precision'] for i in range(len(class_report) - 3)
        }
        recall_per_class = {
            f'label{i}_recall': class_report[str(i)]['recall'] for i in range(len(class_report) - 3)
        }
        result_dict = {
            'accuracy': accuracy,
            'f1': f1,
            'loss': average_loss,
            **precision_per_class,
            **recall_per_class
        }
    else:
        result_dict = {
            'accuracy': accuracy,
            'f1': f1,
            'loss': average_loss
        }

    return result_dict


def watermark_model(model, dataloader, max_epochs=100, threshold=0.9, lr=0.01, momentum=0.2, weight_decay=0,
                    device="cpu", start_acc=0.0, handle_bn=False, eval_loader=None):
    """
    Train a model on watermark data until a target accuracy threshold is reached.

    Args:
        model (torch.nn.Module): The model to watermark.
        dataloader (DataLoader): Loader providing watermark training data.
        max_epochs (int): Maximum number of training epochs.
        threshold (float): Accuracy threshold to stop training.
        lr (float): Learning rate.
        momentum (float): Momentum for SGD.
        weight_decay (float): L2 regularization strength.
        device (str): Device to run training on ('cpu' or 'cuda').
        start_acc (float): Initial accuracy to start from.
        handle_bn (bool): If True, freezes BatchNorm layers during training.
        eval_loader (DataLoader, optional): Evaluation DataLoader, defaults to `dataloader`.

    Returns:
        tuple: (Final accuracy after watermarking, number of epochs used).
    """
    accuracy = start_acc
    epoch = 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.to(device)

    print("Watermarking!")
    while accuracy < threshold:
        model.train()
        if handle_bn:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        print(accuracy)
        cnt = 0
        total_loss = 0
        for data, target in dataloader:
            cnt += 1
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch += 1
        eval_loader = eval_loader or dataloader
        result_dict = evaluate_model(model, eval_loader, device)
        accuracy = result_dict["accuracy"]

        if max_epochs and epoch >= max_epochs:
            break

    return accuracy, epoch

def load_model(mf, model, device="cpu"):
    """
    Load model weights from file into a model instance.

    Args:
        mf (str): Path to the model file.
        model (callable): Model constructor.
        device (str): Device to map the model to ("cpu" or "cuda").

    Returns:
        torch.nn.Module: Loaded model.
    """
    cur_model = model()
    cur_model.load_state_dict(torch.load(mf, map_location=device))
    return cur_model

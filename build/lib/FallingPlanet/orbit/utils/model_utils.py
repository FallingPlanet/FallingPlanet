from torch.utils.tensorboard import SummaryWriter
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef



class MetricsWrapper:
    def __init__(self, num_classes, device, mode='macro'):
        self.mode = mode
        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
        self.precision = Precision(average=self.mode, num_classes=num_classes, task="multiclass").to(device)
        self.recall = Recall(average=self.mode, num_classes=num_classes, task="multiclass").to(device)
        self.f1 = F1Score(average=self.mode, num_classes=num_classes, task="multiclass").to(device)
        self.mcc = MatthewsCorrCoef(num_classes=num_classes, task="multiclass").to(device)
        self.weighted_f1 = F1Score(average='weighted', num_classes=num_classes, task="multiclass").to(device)

    def update(self, outputs, labels):
        self.accuracy.update(outputs, labels)
        self.precision.update(outputs, labels)
        self.recall.update(outputs, labels)
        self.f1.update(outputs, labels)
        self.mcc.update(outputs, labels)
        self.weighted_f1.update(outputs, labels)

    def compute(self):
        accuracy = self.accuracy.compute().item()
        precision = self.precision.compute().item()
        recall = self.recall.compute().item()
        f1 = self.f1.compute().item()
        mcc = self.mcc.compute().item()
        weighted_f1 = self.weighted_f1.compute().item()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "weighted_f1": weighted_f1
        }

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.mcc.reset()
        self.weighted_f1.reset()
        

def log_model_to_tensorboard(model, dummy_input, log_dir="torchlogs/", model_name="model_graph"):
    """
    Logs a PyTorch model to TensorBoard along with a dummy input to properly visualize the model graph.

    Parameters:
    - model: The PyTorch model to log.
    - dummy_input: A tensor or a tuple of tensors that simulate the model's actual input.
    - log_dir: The directory where the TensorBoard logs will be saved.
    - model_name: A name for the model graph, used in TensorBoard.

    This function creates a TensorBoard log file in the specified directory
    that includes the model's computational graph, utilizing the provided dummy input
    for accurate graph representation. This is useful for visualization and debugging purposes.
    """
    writer = SummaryWriter(log_dir)
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Error logging model to TensorBoard with {model_name}: {e}")
    finally:
        writer.close()

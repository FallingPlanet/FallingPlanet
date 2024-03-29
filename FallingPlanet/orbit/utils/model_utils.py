from torch.utils.tensorboard import SummaryWriter
import torch

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

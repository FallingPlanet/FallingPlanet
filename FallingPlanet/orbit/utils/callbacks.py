import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None
        

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0
        return False


class BackupCallback:
    def __init__(self,save_path, save_interval):
        self.save_path = save_path
        self.save_interval = save_interval
        self.step_counter = 0
        
    def step(self,model):
        self.step_counter +=1
        if self.step_counter % self.save_interval == 0:
            torch.save(model.state_dict(),self.save_path.format(step=self.step_counter))
            

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best = None

    def step(self, epoch, model, current_value):
        if self.best is None or current_value < self.best:
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved from {self.best} to {current_value}, saving model to {self.filepath}")
            torch.save(model.state_dict(), self.filepath)
            self.best = current_value
        elif self.verbose:
            print(f"Epoch {epoch}: {self.monitor} did not improve from {self.best}")

    
    
class TerminateOnNaN:
    def __init__(self):
        self.terminated = False

    def step(self, current_value):
        if torch.isnan(current_value):
            print("Training terminated: NaN encountered.")
            self.terminated = True

class save_model_on_exit:
    def __init__(self,model,filepath):
        self.model = model
        self.filepath = filepath
        
    def save_model(self,):
        torch.save(self.model.state_dict(), self.filepath)
        print("Model saved upon exit.")
    
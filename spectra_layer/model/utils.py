import torch

def load_pretrained_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()  # Set the model to inference mode

# gradcam.py
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = list(model.children())[-1]

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        grads = self.gradients.detach()
        acts = self.activations.detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy()

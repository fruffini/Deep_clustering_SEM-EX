import torch
import numpy as np
import os
import h5py

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def directional_derivative(model, cav, layer_name, class_name):
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    return np.dot(gradient, cav) < 0

def flatten_activations_and_get_labels(concepts, layer_name, activations):
    '''
    :param concepts: different name of concepts
    :param layer_name: the name of the layer to compute CAV on
    :param activations: activations with the size of num_concepts * num_layers * num_samples
    :return:
    '''
    # in case of different number of samples for each concept
    min_num_samples = np.min([activations[c][layer_name].shape[0] for c in concepts])
    # flatten the activations and mark the concept label
    data = []
    concept_labels = np.zeros(len(concepts) * min_num_samples)
    for i, c in enumerate(concepts):
        data.extend(activations[c][layer_name][:min_num_samples].reshape(min_num_samples, -1))
        concept_labels[i * min_num_samples : (i + 1) * min_num_samples] = i
    data = np.array(data)
    return data, concept_labels

def get_activations(model, output_dir, data_loader, concept_name, layer_names, max_samples):
    '''
    The function to generate the activations of all layers for ONE concept only
    :param model:
    :param output_dir:
    :param data_loader: the dataloader for the input of ONE concept
    :param layer_names:
    :param max_samples:
    :return:
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.eval()
    activations = {}
    for l in layer_names:
        activations[l] = []

    for i, data in enumerate(data_loader):
        if i == max_samples:
            break
        data = data[0].to(device)
        _ = model(data)
        for l in layer_names:
            z = model.intermediate_activations[l].cpu().detach().numpy()
            activations[l].append(z)

    for l in layer_names:
        activations[l] = np.concatenate(activations[l], axis=0)

    with h5py.File(os.path.join(output_dir, 'activations_%s.h5' % concept_name), 'w') as f:
        for l in layer_names:
            f.create_dataset(l, data=activations[l])


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations
class ModelWrapper(object):
    def __init__(self, model, layers):
        # self.model = deepcopy(model)
        self.model = model
        self.intermediate_activations = {}

        def save_activation(name):
            '''create specific hook by module name'''

            def hook(module, input, output):
                self.intermediate_activations[name] = output

            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(save_activation(name))

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        activation.register_hook(self.save_gradient)
        logit = self.output[:, c]
        logit.backward(torch.ones_like(logit), retain_graph=True)
        # gradients = grad(logit, activation, retain_graph=True)[0]
        # gradients = gradients.cpu().detach().numpy()
        gradients = self.gradients.cpu().detach().numpy()
        return gradients

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, x):
        self.output = self.model(x)
        return self.output
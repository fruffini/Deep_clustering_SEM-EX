import numpy as np
import os
from src.utils.util_CNN import get_activations, load_activations
import torch
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from util_TCAV import flatten_activations_and_get_labels, directional_derivative


class CAV(object):
    def __init__(self, concepts, layer_name, lr, model_type):
        self.concepts = concepts
        self.layer_name = layer_name
        self.lr = lr
        self.model_type = model_type

    def train(self, activations):
        data, labels = flatten_activations_and_get_labels(self.concepts, self.layer_name, activations)

        # default setting is One-Vs-All
        assert self.model_type in ['linear', 'logistic']
        if self.model_type == 'linear':
            model = SGDClassifier(alpha=self.lr)
        else:
            model = LogisticRegression()

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
        model.fit(x_train, y_train)
        '''
        The coef_ attribute is the coefficients in linear regression.
        Suppose y = w0 + w1x1 + w2x2 + ... + wnxn
        Then coef_ = (w0, w1, w2, ..., wn). 
        This is exactly the normal vector for the decision hyperplane
        '''
        if len(model.coef_) == 1:
            self.cav = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cav = -np.array(model.coef_)

    def get_cav(self):
        return self.cav


def tcav_score(model, data_loader, cav, layer_name, class_list, concept, device):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []
    with tqdm(data_loader, unit="batch") as tq_:
        tq_.set_description('Calculating tcav score for %s' % concept)
        for x, _,_,_,_ in tq_:
            model.eval()
            x = x.to(device)
            outputs = model(x)
            k = int(outputs.argmax(axis=1).cpu().detach().numpy())
            if k in class_list:
                derivatives[k].append(directional_derivative(model, cav, layer_name, k))
    score = np.zeros(len(class_list))
    for i, k in enumerate(class_list):
        score[i] = np.array(derivatives[k]).astype(np.int).sum(axis=0) / len(derivatives[k])
    return score


class TCAV(object):
    def __init__(self, model, input_dataloader, concept_dataloaders, class_list, max_samples,cfg):
        self.model = model
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = list(concept_dataloaders.keys())
        self.output_dir = os.path.join(cfg.trained_folder, 'output')
        self.max_samples = max_samples
        self.lr = 1e-3
        self.model_type = 'linear'
        self.class_list = class_list

    def generate_activations(self, layer_names):
        for concept_name, data_loader_c in self.concept_dataloaders.items():
            get_activations(self.model, self.output_dir, data_loader_c, concept_name, layer_names, self.max_samples)

    def load_activations(self):
        self.activations = {}
        for concept_name in self.concepts:
            self.activations[concept_name] = load_activations(
                os.path.join(self.output_dir, 'activations_%s.h5' % concept_name))

    def generate_cavs(self, layer_name):
        cav_trainer = CAV(self.concepts, layer_name, self.lr, self.model_type)
        cav_trainer.train(self.activations)
        self.cavs = cav_trainer.get_cav()

    def calculate_tcav_score(self, layer_name, output_path):
        self.scores = np.zeros((self.cavs.shape[0], len(self.class_list)))
        for i, cav in enumerate(self.cavs):
            self.scores[i] = tcav_score(self.model, self.input_dataloader, cav, layer_name, self.class_list,
                                        self.concepts[i])
        # print(self.scores)
        np.save(output_path, self.scores)
"""
This file is exactly the same as the "contextualized_topic_models" package - inference_network.py

You can choose to "from contextualized_topic_models.networks.inference_network import ContextualInferenceNetwork"

PyTorch class for feed forward inference network.
"""

from collections import OrderedDict
from torch import nn
import torch


class ContextualInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, vocab_size, latent_representation_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        """
        Initialize InferenceNetwork.

        Args
            vocab_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(ContextualInferenceNetwork, self).__init__()
        assert isinstance(vocab_size, int), "vocab_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(vocab_size+vocab_size, hidden_sizes[0])
        self.adapt = nn.Linear(latent_representation_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, latent_representation):
        """Generate mu and sigma - Inference Network"""
        latent_representation = self.adapt(latent_representation)

        x = self.activation(latent_representation)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma




class CombinedInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, vocab_size, latent_representation_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        """
        Initialize InferenceNetwork.

        Args
            vocab_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(CombinedInferenceNetwork, self).__init__()
        assert isinstance(vocab_size, int), "vocab_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(vocab_size+vocab_size, hidden_sizes[0])
        self.adapt = nn.Linear(latent_representation_size, vocab_size)
        self.bert_layer = nn.Linear(hidden_sizes[0], hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, latent_representation):
        """Forward pass."""
        latent_representation = self.adapt(latent_representation)
        x = torch.cat((x, latent_representation), 1)
        x = self.input_layer(x)

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

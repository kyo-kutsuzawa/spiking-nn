import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
from Release.snn import (  # type: ignore
    DoubleExponentialSynapticFilter,
    IzhikevichNeuron,
    SpikingNeuralNetwork,
)

# class SpikingNN:
#     """Spiking neural network (SNN) with FORCE learning.

#     This SNN consists of
#     1) neurons with the Izhikevich model and
#     2) synapses with the double exponential synaptic filter.
#     """

#     def __init__(self, n_units: int, in_size: int, out_size: int):
#         """Initialization.

#         Parameters
#         ----------
#         n_units (int):
#         n_units: int
#             Number of neurons.
#         in_size: int
#             Input size.
#         out_size: int
#             Output size.
#         """

#         self.neurons = IzhikevichNeuron(n_units=n_units)
#         self.synapses = DoubleExponentialSynapticFilter(n_units=n_units)

#         self.p = 0.1  # degree of sparsity in the network
#         self.G = 5e3  # scale of the static weight matrix
#         self.Q = 5e3  # scale of the feedback term
#         self.eta = np.random.uniform(
#             -1.0, 1.0, size=(n_units, out_size)
#         )  # encoder that contributes to the tuning preferences of the neurons in the network
#         self.w0 = np.random.normal(
#             0, 1 / (np.sqrt(n_units) * self.p), size=(n_units, n_units)
#         )  # sparse and static weight matrix
#         self.phi = np.zeros((n_units, out_size))  # decoder that is determined by RLS.
#         self.i_bias = 1000.0  # bias current

#         self.mask = np.where(
#             np.random.uniform(0, 1, size=(n_units, n_units)) < self.p, 1, 0
#         )
#         self.w0 *= self.mask

#         l = 2.0  # regularization parameter
#         self.P = np.identity(n_units) / l  # used for RLS

#         self.n_units = n_units

#         self.Gw0 = self.G * self.w0.T
#         self.Qeta = self.Q * self.eta.T

#     def reset_state(self):
#         self.neurons.reset_state()
#         self.synapses.reset_state()
#         self.x: np.ndarray = np.dot(self.synapses.r, self.phi)

#     def update(self):
#         # Calculate input currents
#         s = np.dot(self.synapses.r, self.Gw0) + np.dot(self.x, self.Qeta)
#         i = s + self.i_bias

#         # Update the states of neurons and synapses
#         spikes = self.neurons.update(i)
#         self.synapses.update(spikes)

#         # Calculate the output, x
#         self.x = np.dot(self.synapses.r, self.phi)
#         return self.x

#     def train(self, teaching_signal: np.ndarray):
#         err = self.x - teaching_signal

#         # Update P
#         Pr: np.ndarray = self.P.dot(self.synapses.r.T).reshape((-1, 1))
#         rPr: np.ndarray = self.synapses.r.dot(Pr)
#         c = 1.0 / (1.0 + rPr)
#         self.P -= Pr.dot(Pr.T) * c

#         # Update phi
#         self.phi -= err * Pr

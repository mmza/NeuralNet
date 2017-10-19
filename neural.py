#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:25:56 2017

@author: mike
"""

import numpy as np

class Perceptron:

    def __init__(self, w, b):

        self.w = w
        self.b = b

    def forward_pass(self, single_input):

        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b
        
        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):        
        return (np.dot(input_matrix, self.w) + self.b) > 0
    
    def train_on_single_example(self, example, y):
        prediction = float(example.T.dot(self.w) + self.b > 0)        
        error = y - prediction        
        delta = error*example        
        self.w = self.w + delta        
        self.b += error

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    
    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):

        assert weights.shape[1] == 1, "Incorrect weight shape"
        
        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        

        
    def forward_pass(self, single_input):
        
        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)
    
    def summatory(self, input_matrix):
        
        return np.dot(input_matrix, self.w)
    
    def activation(self, summatory_activation):
        
        return np.array([sigmoid(summatory_activation[i]) for i in range(len(summatory_activation))])
    
    def vectorized_forward_pass(self, input_matrix):
        
        return self.activation(self.summatory(input_matrix))
    
    def J_quadratic(neuron, X, y):
        
        assert y.shape[1] == 1, 'Incorrect y shape'    
        return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)
    
    def J_quadratic_derivative(y, y_hat):
        
        assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'    
        return (y_hat - y) / len(y)
    
    def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
        
        z = neuron.summatory(X)
        y_hat = neuron.activation(z)

        dy_dyhat = J_prime(y, y_hat)
        dyhat_dz = neuron.activation_function_derivative(z)        
        dz_dw = X
        
        grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)
        grad = grad.T       
        return grad
    
    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        
        indices = np.arange(len(X))        
        def get_new_batch():
            batch_indices = np.random.choice(indices, batch_size, replace = False)
            return (X[batch_indices, :], y[batch_indices, :])    
        for _ in range(max_steps):
            upd = self.update_mini_batch(*get_new_batch(), learning_rate, eps)
            if upd: return 1
        return 0
    
    def update_mini_batch(self, X, y, learning_rate, eps):
        
        init_J = J_quadratic(self, X, y)
        gr = compute_grad_analytically(self, X, y)
        self.w = self.w - learning_rate*gr
        new_J = J_quadratic(self, X, y)
        return int(init_J - new_J < eps)

    
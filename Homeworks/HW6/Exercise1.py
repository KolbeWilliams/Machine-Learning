#Exercise 1:
from turtle import right
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def likelihood(x, y, model):
    return (factorial(x + (y - x)) / (factorial(x) * factorial((x + (y - x)) - x))) * (model**x) * ((1 - model)**(y - x))

def post_prob(x, y, model_arr, prior_arr):
    post_prob_arr = np.array([])
    likelihoods = np.array([likelihood(x, y, model) for model in model_arr])
    post_prob_arr = likelihoods * prior_arr
    return post_prob_arr / np.sum(post_prob_arr)

model = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
uniform_priorProb = np.full([11], 0.1)
right_skewed_priorProb = np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.001, 0.0])
bimodal_priorProb = np.array([0.7, 0.5, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3, 0.5, 0.7])
mound_shaped_priorProb = np.array([0.05, 0.10, 0.16, 0.19, 0.18, 0.15, 0.09, 0.05, 0.02, 0.01, 0.00])

fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[0, 0].plot(model, uniform_priorProb, c = 'r', label = 'prior')
ax[0, 0].plot(model, post_prob(3, 5, model, uniform_priorProb), label = 'posterior(3/5)', linestyle = '-.', c = 'b')
ax[0, 0].plot(model, post_prob(15, 25, model, uniform_priorProb), label = 'posterior(15/25)', linestyle = '--', c = 'b')
ax[0, 0].plot(model, post_prob(75, 125, model, uniform_priorProb), label = 'posterior(75/125)', c = 'b')
ax[0, 0].set_title('Uniform prior distribution')
ax[0, 0].set_xlabel('model(value of p)')
ax[0, 0].legend()

ax[0, 1].plot(model, right_skewed_priorProb, c = 'r', label = 'prior')
ax[0, 1].plot(model, post_prob(3, 5, model, right_skewed_priorProb), label = 'posterior(3/5)', linestyle = '-.', c = 'b')
ax[0, 1].plot(model, post_prob(15, 25, model, right_skewed_priorProb), label = 'posterior(15/25)', linestyle = '--', c = 'b')
ax[0, 1].plot(model, post_prob(75, 125, model, right_skewed_priorProb), label = 'posterior(75/125)', c = 'b')
ax[0, 1].set_title('Right-skewed prior distribution')
ax[0, 1].set_xlabel('model(value of p)')
ax[0, 1].legend()

ax[1, 0].plot(model, bimodal_priorProb, c = 'r', label = 'prior')
ax[1, 0].plot(model, post_prob(3, 5, model, bimodal_priorProb), label = 'posterior(3/5)', linestyle = '-.', c = 'b')
ax[1, 0].plot(model, post_prob(15, 25, model, bimodal_priorProb), label = 'posterior(15/25)', linestyle = '--', c = 'b')
ax[1, 0].plot(model, post_prob(75, 125, model, bimodal_priorProb), label = 'posterior(75/125)', c = 'b')
ax[1, 0].set_title('Bimodal prior distribution')
ax[1, 0].set_xlabel('model(value of p)')
ax[1, 0].legend(loc = 'upper right')

ax[1, 1].plot(model, mound_shaped_priorProb, c = 'r', label = 'prior')
ax[1, 1].plot(model, post_prob(3, 5, model, mound_shaped_priorProb), label = 'posterior(3/5)', linestyle = '-.', c = 'b')
ax[1, 1].plot(model, post_prob(15, 25, model, mound_shaped_priorProb), label = 'posterior(15/25)', linestyle = '--', c = 'b')
ax[1, 1].plot(model, post_prob(75, 125, model, mound_shaped_priorProb), label = 'posterior(75/125)', c = 'b')
ax[1, 1].set_title('Mound-shaped prior distribution')
ax[1, 1].set_xlabel('model(value of p)')
ax[1, 1].legend()
plt.tight_layout()
plt.show()
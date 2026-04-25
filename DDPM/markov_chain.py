""" so where do i get started with this ?
we're implementing a markov chain
what it means in very layman terms is that a state x depends purely on its prev state x-1
ofc there's more to it, but this is what it means in very simple terms

So firstly, there are states, for each state, we create a NxN transition matrix
the values that go inside the matrix are the probabilities of a state transition happening 
"""
import numpy as np
states = [0,1]
transition_matrix = np.array([[0.8,0.2],
                             [0.3,0.7]
                            ])

def next_st(current):
    return np.random.choice([0,1],p = transition_matrix[current])
state = 0
trajectory = []
for _ in range(100):
    state = next_st(state)
    trajectory.append(state)

print(np.mean(trajectory))

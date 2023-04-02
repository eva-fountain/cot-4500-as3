# Eva Fountain
# COT4500, Spring 2023
# Assignment 3

import numpy as np

# Number 1

np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function(t: float, y: float)
    return t - (y**2)

def do_work(t, y, h):
    basic_function_call = function(t, y)

    incremented_t = t + y
    incremented_y = y + (h * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_y)

    return basic_function_call + incremented_function_call

def modified_eulers():
    original_y = .5
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # Set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # Do we have all values ready?
        t = start_of_t
        y = original_y
        h = h

        # Create a function for the inner work
        inner_math = do_work(t, y, h)

        # Next approximation
        next_y = y + ((h / 2) * inner_math)

        print(next_y)

        # Set just solved "y" to original y
        # and change t
        start_of_t = t + h
        original_y = next_y

    return none

if __name__ == "__main__":
    modified_eulers()
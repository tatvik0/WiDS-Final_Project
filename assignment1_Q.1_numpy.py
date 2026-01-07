import numpy as np

# Create the 5x5 random array

np.random.seed(0)  # for reproducible output
arr = np.random.randint(1, 101, size=(5, 5))  # random ints 1–100
print("Original 5x5 array:\n", arr)


# (a) Middle element

# For a 5x5 array, middle index is (2, 2) in 0-based indexing
middle_element = arr[2, 2]
print("\n(a) Middle element of the array:", middle_element)


# (b) Mean of each row

row_means = np.mean(arr, axis=1)
print("\n(b) Mean of each row:", row_means)

# (c) Elements > overall mean

overall_mean = np.mean(arr)       # mean of all elements
print("\n(c) Overall mean of the array:", overall_mean)

mask = arr > overall_mean         # boolean mask
greater_than_mean = arr[mask]
print("Elements greater than overall mean:", greater_than_mean)


# (d) Spiral order function

def numpy_spiral_order(matrix):
    """
    Take a 2-D NumPy array and return a list of
    its elements in clockwise spiral order.
    """
    rows, cols = matrix.shape
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    result = []

    # standard iterative spiral traversal
    while top <= bottom and left <= right:

        # traverse from left to right across the top row
        for c in range(left, right + 1):
            result.append(matrix[top, c])
        top += 1

        # traverse from top to bottom down the right column
        for r in range(top, bottom + 1):
            result.append(matrix[r, right])
        right -= 1

        # traverse from right to left across the bottom row
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom, c])
            bottom -= 1

        # traverse from bottom to top up the left column
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r, left])
            left += 1

    return result

spiral_list = numpy_spiral_order(arr)
print("\n(d) Spiral order:", spiral_list)


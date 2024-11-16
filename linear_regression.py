import numpy as np

def poly_feature_transform(X, poly_order=1):
    """Transforms the input data X to match the specified polynomial order.

    Arguments:
        X {np.ndarray} -- A numpy array of shape (N, D) containing N instances
        with D features.
        poly_order {int} -- Order of polynomial of the hypothesis function

    Returns:
        np.ndarray -- A numpy array of shape (N, (D * order) + 1) representing
        the transformed features following the specified `poly_order`
    """

    # TODO: Add features to X until poly_order
    # Ensure X is 2D

    if X.ndim == 1:
        X = X.reshape(-1, 1)  # Reshape to (N, 1)

    N, D = X.shape

    # Initialize the transformed features array with ones for bias term
    f_transform = np.ones((N, D + 1))

    # Add original features to the transformed array
    f_transform[:, 0:D] = X  # Fill in the original features

    # If poly_order is greater than 1, create additional polynomial features
    if poly_order > 1:
        for i in range(2, poly_order + 1):
            for j in range(D):
                f_transform = np.column_stack((f_transform, X[:, j] ** i))

    return f_transform
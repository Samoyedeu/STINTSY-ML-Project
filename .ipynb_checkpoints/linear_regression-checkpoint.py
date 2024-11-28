import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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

def compute_RMSE(y_true, y_pred):
    # TODO: Compute the Root Mean Squared Error
    rmse = np.sqrt( np.square(np.subtract(y_true, y_pred)).mean() )

    return rmse

def compute_R2_Score(y_true, y_pred):
    # Compute the mean of the true values
    mean_true = sum(y_true) / len(y_true)

    # Compute Total Sum of Squares
    tss = sum((yi - mean_true) ** 2 for yi in y_true)

    # Compute Residual Sum of Squares
    rss = sum((yi - yhat) ** 2 for yi, yhat in zip(y_true, y_pred))

    # Compute R-squared
    r2_score = 1 - rss / tss

    return r2_score

def compute_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f" Metrics for {model_name}:")
    print(f" Mean Squared Error (MSE): {mse:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" R-squared (R²): {r2:.4f}")
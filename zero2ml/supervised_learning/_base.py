from zero2ml.utils.evaluation_metrics import Accuracy, RSquared

class BaseModel:
    """
    Base class for all supervised learning models in zero2ml.

    Models should specify ``self.model_type`` as one of ["classifier", "regressor"].
    """
    def score(self, X, y):
        """
        Calculate mean accuracy (for classification)
        or coefficient of determination (for regression)
        of predictions on a given data.
        Parameters
        ----------
        X: array_like (m, n)
            Features dataset with shape m examples and n features.
        y: array_like(m,)
            Target dataset with m examples.
        Returns
        -------
        Mean accuracy or coefficient of determination of predictions.
        """
        # Make predictions with the trained model
        y_pred = self.predict(X)

        if self.model_type == "classifier":

            # Calculate accuracy
            accuracy = Accuracy()
            score_value = accuracy(y_pred, y)

        if self.model_type == "regressor":

            # Calculate R^2
            R2 = RSquared()
            score_value = R2(y_pred, y)

        return score_value

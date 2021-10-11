import numpy
from sklearn import neighbors
from sklearn.neighbors import (KNeighborsClassifier, NearestNeighbors,
                               KNeighborsRegressor)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist as scipy_cdist

from tslearn.metrics import cdist_dtw, cdist_soft_dtw, \
    cdist_sax, TSLEARN_VALID_METRICS
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.utils import (to_time_series_dataset, to_sklearn_dataset,
                           check_dims)
from tslearn.bases import BaseModelPackage

neighbors.VALID_METRICS['brute'].extend(['dtw', 'softdtw', 'sax'])


class KNeighborsTimeSeriesClassifier(BaseModelPackage,
                                     KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote for Time Series.
    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:
        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`
    class (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the 
        creation of the `SymbolicAggregateApproximation` object.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.
    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.
    Examples
    --------
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      metric_params={
    ...                                          "itakura_max_slope": 2.},
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    """  # noqa: E501
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None,
                 verbose=0):
        KNeighborsClassifier.__init__(self,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _is_fitted(self):
        check_is_fitted(self, '_ts_fit')
        return True

    def _get_model_params(self):
        return {
            '_X_fit': self._X_fit,
            '_ts_fit': self._ts_fit,
            '_d': self._d,
            'classes_': self.classes_,
            '_y': self._y,
            'outputs_2d_': self.outputs_2d_
        }

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, )
            Target values.
        Returns
        -------
        KNeighborsTimeSeriesClassifier
            The fitted estimator
        """
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            if self._ts_metric == 'sax':
                if self.metric_params is not None:
                    self._ts_fit = self._sax_preprocess(X,
                                                        **self.metric_params)
                else:
                    self._ts_fit = self._sax_preprocess(X)

            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        Returns
        -------
        array, shape = (n_ts, )
            Array of predicted class labels
        """
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict(X_)

    def predict_proba(self, X):
        """Predict the class probabilities for the provided data
        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        Returns
        -------
        array, shape = (n_ts, n_classes)
            Array of predicted class probabilities
        """
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict_proba(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict_proba(X_)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}

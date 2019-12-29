# author: Gael Varoquaux <gael.varoquaux@inria.fr>
# License: BSD 3 clause
# Copyright: INRIA

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt


# #############################################################################
# Generate the data


def generate_data(n_samples: int = 60,
                  n_features: int = 20,
                  random_state: int = 1,
                  alpha: float = .98,
                  smallest_coef: float = .4,
                  largest_coef: float = .7,
                  ):
    prng = np.random.RandomState(random_state)
    prec = make_sparse_spd_matrix(n_features,
                                  alpha=alpha,
                                  smallest_coef=smallest_coef,
                                  largest_coef=largest_coef,
                                  random_state=prng)
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    emp_cov = np.dot(X.T, X) / n_samples
    return prec, cov, X, emp_cov


# #############################################################################
# Estimate the covariance


def demo():
    prec, cov, X, emp_cov = generate_data()

    model = GraphicalLassoCV(cv=5)
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_

    lw_cov_, _ = ledoit_wolf(X)
    lw_prec_ = linalg.inv(lw_cov_)

    # #############################################################################
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)

    # plot the covariances
    covs = [('Empirical', emp_cov), ('Ledoit-Wolf', lw_cov_),
            ('GraphicalLassoCV', cov_), ('True', cov)]
    vmax = cov_.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 4, i + 1)
        plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)

    # plot the precisions
    precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
             ('GraphicalLasso', prec_), ('True', prec)]
    vmax = .9 * prec_.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 4, i + 5)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s precision' % name)
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor('.7')
        else:
            ax.set_axis_bgcolor('.7')

    # plot the model selection metric
    plt.figure(figsize=(4, 3))
    plt.axes([.2, .15, .75, .7])
    plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
    plt.axvline(model.alpha_, color='.5')
    plt.title('Model selection')
    plt.ylabel('Cross-validation score')
    plt.xlabel('alpha')

    plt.show()

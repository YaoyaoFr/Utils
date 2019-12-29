import os
import numpy as np
import tensorflow as tf
from numpy.linalg import pinv, lstsq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def standard(x: np.ndarray,
             mean: bool = True,
             std: bool = True,
             ) -> np.ndarray:
    """
    x_hat = x - x_bar
    :param x: np.ndarray with size [p, N] where p is the number of variables and N is the number of observations.
    :param mean:
    :param std:
    :return: x_hat
    """
    result = {'x': x}
    p, N = np.shape(x)

    if mean:
        x_mean = np.reshape(np.mean(x, axis=1),
                            newshape=[p, 1])
        result['x_mean'] = x_mean
        E = np.ones(shape=[1, N])
        x_mean = np.matmul(x_mean, E)
        x_hat = x - x_mean
        if std:
            E = np.ones(shape=[1, N])
            x_std = np.reshape(np.std(x, axis=1), newshape=[p, 1])
            result['x_std'] = x_std

            x_std = np.matmul(x_std, E)
            x_hat = x_hat / x_std
        result['x_hat'] = x_hat

    return result


def self_covariance(x: np.ndarray) -> np.ndarray:
    """
    C = x * x_T
    :param x: np.ndarray with size [p, N] where p is the number of variables and N is the number of observations.
    :return: C
    """
    x = standard(x, std=False)
    x_T = x.T
    C = np.matmul(x, x_T)
    return C


def covariance(x: np.ndarray,
               y: np.ndarray) -> np.ndarray:
    """
    C = x * y
    :param x: np.ndarray with size [p, N] where p is the number of variables and N is the number of observations.
    :param y: np.ndarray with size [q, N] where p is the number of variables and N is the number of observations.
    :return: C with size [p, q]
    """
    x = standard(x, std=False)
    y = standard(y, std=False)
    y_T = y.T
    C = np.matmul(x, y_T)
    return C


def regression(x, y):
    """

        :param x: with shape [q, N]
        :param y:            [p, N]
        :param corr_type: subset in ['partial', 'part']
        :return:
        """
    if len(np.shape(x)) == 1:
        N,  = np.shape(x)
        x = np.reshape(x, newshape=[1, N])
    if len(np.shape(y)) == 1:
        N,  = np.shape(y)
        y = np.reshape(y, newshape=[1, N])

    E = np.ones(shape=[1, N])

    y_aug = np.concatenate((y, E),
                           axis=0)
    Beta = lstsq(y_aug.T, x.T)[0].T
    res = x - np.matmul(Beta, y_aug)

    return {'residual': res,
            'Beta': Beta,
            }


def partial_corr(y: np.ndarray,
                 x: np.ndarray,
                 corr_type: list = ['part'],
                 ):
    """

    :param y: with shape [p, N]
    :param x:            [k, N]
    :param corr_type: subset in ['partial', 'part']
    :return:
    """
    p, N = np.shape(y)
    k, _ = np.shape(x)

    E = np.ones(shape=[1, N])
    x_k = np.reshape(x[-1, :], newshape=[1, N])
    x_k_1_aug = np.concatenate((x[:k, :], E), axis=0)

    Beta = lstsq(x_k_1_aug.T, y.T)[0]

    stan_x = standard(y)
    stan_y = standard(x)
    beta = lstsq(stan_y['x_hat'].T, stan_x['x_hat'].T)[0]

    result = {'Beta': Beta,
              'beta': beta,
              'x_mean': stan_x['x_mean'],
              'x_std': stan_x['x_std'],
              'y_mean': stan_y['x_mean'],
              'y_std': stan_y['x_std'],
              }
    for corr in corr_type:
        if corr == 'partial':
            x_hat = y - np.matmul(Beta.T, x_k_1_aug)
            corr_partial = np.corrcoef(x_hat)
            result['corr_partial'] = corr_partial
        elif corr == 'part':
            x_hat = y - np.matmul(Beta.T, x_k_1_aug)
            corr_part = np.corrcoef(y, x_hat)
            result['corr_part'] = corr_part

    return result


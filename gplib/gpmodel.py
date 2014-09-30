import numpy as np
import scipy.linalg
import logging
import time
logging.basicConfig(level=logging.INFO)

# This class is strongly influenced by Tom Evan's GP module
# https://github.com/tomevans/gps

PERTURB = 1e-10


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        msg = '%r %2.2f sec'
        msg = msg % (method.__name__, te-ts)
        #print msg
        logging.debug(msg)
        return result
    return timed

class GPModel:
    _mu = None
    _cov = None
    _kn = None
    _knp = None
    _kp = None
    _mtrain = None
    _rtrain = None
    _mnew = None
    _log_likelihood = None

    def __init__(self, x, y, f_mean, f_covariance, f_error, hyperparameters,
                 hyperconstraint=None):
        """
        Initialize the GP object with the input and output arrays.
        """
        if hyperconstraint is None:
            hyperconstraint = GPModel.pass_through()
        self.hyperconstraint = hyperconstraint
        n = len(x)
        self.xtrain = np.reshape(x, (n, 1))  # Input
        self.dtrain = np.reshape(y, (n, 1))  # Output
        self.mfunc = f_mean
        self.cfunc = f_covariance
        self.efunc = f_error
        self.hyperparameters = hyperparameters
        self.cpars = self.hyperparameters.copy()  # Combined parameters

    @staticmethod
    def pass_through(*_):
        return 1.0

    @property
    def mu(self):
        if self._mu is None:
            self.meancov(self.xtrain)
        return self._mu

    @property
    def cov(self):
        if self._cov is None:
            self.meancov(self.xtrain)
        return self._cov

    @property
    def kn(self):
        if self._kn is None:
            self.meancov(self.xtrain)
        return self._kn

    @property
    def knp(self):
        if self._knp is None:
            self.meancov(self.xtrain)
        return self._knp

    @property
    def mtrain(self):
        if self._mtrain is None:
            self.meancov(self.xtrain)
        return self._mtrain

    @property
    def rtrain(self):
        if self._rtrain is None:
            self.meancov(self.xtrain)
        return self._rtrain

    @property
    def mnew(self):
        if self._mnew is None:
            self.meancov(self.xtrain)
        return self._mnew

    @timeit
    def meancov(self, xnew):
        """
        Returns the mean and full covariance of a gp at the locations of xnew

        Example
        -------
        mu, cov, kn = meancov(xnew)

        Returns
        -------
        
        'mu' [Px1 array] - gp mean function values.
        'cov' [PxP array] - gp covariance values.
        """

        # Timings for this module show that
        # the mean function is super fast, taking
        # ~0.8ms, whereas the cfuncs take 5-10ms each
        # And the lu_solves dominate the time,
        # If reduce cfunc to 0 ms we could get a 50% speed up
        ts = []
        ts.append(time.time())
        mnew = self.mfunc(xnew, **self.cpars).flatten()
        ts.append(time.time())
        
        

        # The number of predictive points:
        p = np.shape(xnew)[0]

        # Evaluate the covariance matrix block for the new points.
        # We do not include an error term and assume this is folded
        # into the covariance matrix
        kp = self.cfunc(xnew, xnew, **self.cpars)
        ts.append(time.time())
        self.efunc(kp, **self.cpars)
        # Add numerical noise along the diagonal
        kp += PERTURB **2.0 * np.eye(kp.shape[0])
        kp = np.matrix(kp)
        ts.append(time.time())

        # Evaluate the mean and covariance and condition on the data
        mtrain = self.mfunc(self.xtrain, **self.cpars)
        rtrain = np.matrix(self.dtrain.flatten() - mtrain.flatten()).T
        mnew = np.matrix(mnew).T
        self._mtrain = mtrain
        self._rtrain = rtrain
        ts.append(time.time())

        # if np.allclose(self.xtrain, xnew, 1e-5):
        #     kn = kp.copy()
        #     knp = kp.copy()
        # else:
        # The covariance within the training set
        kn = self.cfunc(self.xtrain, self.xtrain, **self.cpars)
        # Add in the error model to the points
        self.efunc(kn, **self.cpars)
        # Add numerical noise along the diagonal
        kn += PERTURB **2.0 * np.eye(kn.shape[0])
        # The covariance with the training set and new data
        knp = self.cfunc(self.xtrain,        xnew, **self.cpars)
        kn = np.matrix(kn)
        knp = np.matrix(knp)
        ts.append(time.time())

        # Use Cholesky decompositions to efficiently calculate the mean:
        l = np.linalg.cholesky(kn)
        ts.append(time.time())
        # Inverse covariance of the training data
        kninv_rtrain = scipy.linalg.lu_solve(scipy.linalg.lu_factor(kn), rtrain)
        kninv_rtrain = np.matrix(kninv_rtrain)
        # Predictive mean computed on the new data
        mu = np.array(mnew + knp.T * kninv_rtrain).flatten()
        ts.append(time.time())

        # Now do similar for the covariance matrix:
        linv_knp = scipy.linalg.lu_solve(scipy.linalg.lu_factor(l), knp)
        linv_knp = np.matrix(linv_knp)
        knpt_llinv_knp = linv_knp.T * linv_knp
        cov = np.array(kp - knpt_llinv_knp)
        mu = np.reshape(mu, (p, 1))
        ts.append(time.time())
        self._mu = mu
        self._cov = cov
        self._kn = kn
        self._knp = knp
        self._kp = kp
        self._mtrain = mtrain
        self._rtrain = rtrain
        self._mnew = mnew
        #msg = str((np.diff(ts)*1000.0).astype('int'))
        #logging.debug(msg)
        return mu, cov

    @property
    def log_likelihood(self):
        if self._log_likelihood is not None:
            return self._log_likelihood
        mtrain, kn = self.mtrain, self.kn
        # Evaluate the log likelihood
        n = self.xtrain.shape[0]
        resids = self.dtrain.flatten() - mtrain.flatten()
        resids = np.reshape(resids, (n, 1))
        r = np.matrix(resids)

        # Get the log determinant of the covariance matrix:
        sign, logdet_kn = np.linalg.slogdet(kn)

        # Calculate the product inv(c)*deldm using LU factorisations:
        invkn_r = scipy.linalg.lu_solve(scipy.linalg.lu_factor(kn), r)
        rt_invkn_r = float(r.T * np.matrix(invkn_r))

        # Calculate the log likelihood:
        log_likelihood = - 0.5*logdet_kn - 0.5*rt_invkn_r - 0.5*n*np.log(2*np.pi)

        self._log_likelihood = log_likelihood
        return log_likelihood

    def predict(self, xnew):
        mu_new, cov_new = self.meancov(xnew)
        sig = np.sqrt(np.diag(cov_new))
        mu_new = mu_new.flatten()
        cov_new = cov_new.flatten()
        return mu_new, sig

    @timeit
    def log_posterior(self):
        """ Posterior is simply log prior + log likelihood"""
        # Ensure a valid hyperparameter space; effectively a hyperprior
        # but if the prior is inf no need to evaluate the likelihood!
        # Timing for this method is dominated (75ms) by the log_likelihood func
        hpv = np.array(self.hyperparameters.values())
        if ~self.hyperconstraint(hpv):
            return -np.inf
        log_prior = np.log(hpv).sum()
        if ~np.isfinite(log_prior):
            return -np.inf
        try:
            log_l = self.log_likelihood
        except np.linalg.LinAlgError:
            # We get linalg errors when the matrix isn't positive definite
            # Which may occur if the diagonal elements are too small
            print 'linalg error ', hpv
            return -np.inf
        log_p = log_prior + log_l
        if np.isnan(log_p):
            hps = str(self.hyperparameters)
            logging.warning("Posterior is NAN at %s" % hps)
            return -np.inf
        return log_p

    def neg_log_posterior(self):
        """ 
        Negate the posterior distribution for use with a function 
        optimizer / MAP estimator
        """
        return -self.log_posterior()


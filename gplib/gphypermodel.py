import multiprocessing
import emcee
import numpy as np
from scipy.optimize import basinhopping, fmin
import logging
import pickle
import time
logging.basicConfig(level=logging.DEBUG)
start = time.time()
from gpmodel import GPModel

def fprior(x):
    ret = -np.log(x)
    return -np.inf if np.any(np.isnan(ret)) else ret


class GPHyperModel:
    xtrain = None
    dtrain = None
    fc = None
    fm = None
    fe = None
    initial_parameters = None
    used_initial_parameters = None
    log_posterior = None
    model_instance = None
    mcmc_sampler = None
    map_estimate = None
    marginal_estimate = None
    __name__ = "GPHyperModel"

    def __init__(self, func_mean, func_covariance, func_error, hyperparameter_keys=None,
                 mode='mcmc', model=GPModel, func_hyperconstraint=None, fixed_mean=False,
                 fixed_covariance=False, seed=10):
        """ 
        The Gaussian Process hyper model class is designed to scan
        over many hyperparameters and find the optimal set of GP model
        hyperparameters. The hyperparameter space can be efficiently explored
        via mode='mcmc' or we can find a point estimate by maximizing the 
        the posterior distribution with mode='map'. For each evaluation of the 
        hyperparameters a GPModel class is instantiated and the log posterior
        is evaluated with the given hyperparameters. 

        Parameters
        ----------
        func_mean : function with signature f(X,Y, **hyperparameters)
           func_mean should be a function that, given X, Y and the 
           hyperparameters generates a function that will calculate
           the predictive mean on unseen data. Note that the input
           array X may in general by multidimensional, but Y is
           1D . For example:
           >>> mean = func_mean(X,Y, **hyperparameters)
           >>> Y_mean = mean(X_new)
           >>> assert Y_mean.shape[0] == X.shape[0]

        func_covariance : function with signature f(X0, X1, **hyperparameters)
            func_covariance is similar to func_mean in that it will generate
            a function that in turn evaluates the covariance on new data. 
            Note that the input X is in general multidimensional, and that
            X0 and X1 need not have the same shape but must have the same 
            dimensionality. If the X inputs are 1D, the resulting covariace 
            kernel is 2D.
            >>> cov = func_covariance(X, X, **hyperparameters)
            >>> cmatrix = cov(X, X_new)
         
        Attributes
        ----------
        hyperparameter_keys: list of strings
            hyperparemter_keys is a list of keyword arguments that func_mean
            and func_covariance accepts as hyperparameters.            

        mode : 'map' or 'mcmc'
            Explore the hyperparameter space either by using MCMC or by finding 
            a point estimate by maximizing the negative posterior distribution. 
            The MCMC methodology is fueled by the emcee library. The maximum
            a posteriori (MAP) estimate is estimated by using a combined 
            basinhopping method (to avoid local minimuma) with Nelder-Mead 
            optimization to fine tune the local maximum.

        func_hyperconstraint : function with signature f(**hyperparams) = bool
            func_hyperconstraint allows us to quickly throw out hyperparameters
            that are invalid without having to build the whole model. It simply 
            shortcuts the posterior calculation by returning -np.inf where the
            func_hyperconstraint evaluates to False. This is useful for setting 
            constraints, for example the following constrains hyperparameters
            to the set of positive reals, and equivalent to modifying the
            prior to be negative infinity:
            >>> hyperconstraint = lambda *x: -np.inf if any(x < 0) else 1.0

        fixed_mean : bool
        fixed_covariance : bool
            If func_mean / func_covariance does not vary with a change with the
            hyperparameters, then setting this option allows us to skip 
            re-evaluating them with every change of the hyperparameters.

        Examples
        --------
        >>> hm = GPHyperModel(func_mean, func_covariance, mode='mcmc')
        >>> optimal_model = hm.fit(X, Y, {'holidays':[1,42,89,90]})
        >>> Ynew = hm.predict(Xnew)
        >>> logp = hm.score()
        >>> rms = hm.score('rmse')
        >>> rms_test = hm.score(X_test, Y_test, 'rmse')
        """
        if hyperparameter_keys is not None:
            self.keys = list(hyperparameter_keys) 
        else:
            self.keys = []
        self.func_mean = func_mean
        self.func_covariance = func_covariance
        self.func_error = func_error
        self.fixed_covariance = fixed_covariance
        self.fixed_mean = fixed_mean
        self.mode = mode.lower()
        self.model = model
        if func_hyperconstraint is None:
            # If we do not defined any constraints, pass through
            self.func_hyperconstraint = lambda *x, **y: 1.0
        else:
            self.func_hyperconstraint = func_hyperconstraint
        self.seed = seed
        np.random.seed(self.seed)
        self.logger = logging.getLogger(self.__name__)

        # Only support the following evaluation models
        assert self.mode in ['mcmc', 'map']

    def vectorize_args(self, kwargs):
        """ Convert a dict of args to a numpy array """
        # The arguments must be in vector form for scipy optimize
        vec = [kwargs.pop(k) for k in self.keys]
        assert len(kwargs) == 0  # Must always vectorize all args
        return vec

    def devectorize_args(self, args):
        """ Convert an array of numbers back into a dictionary """
        return {k: v for k, v in zip(self.keys, args)}

    def fit(self, x, y, initial_parameters=None, **kwargs):
        """
        Fit the Gaussian Process model. 

        Returns itself.
        """
        assert x.shape[-1] == y.shape[-1]
        self.xtrain = x
        self.dtrain = y
        self.initial_parameters = initial_parameters
        if self.mode == 'mcmc':
            self.run_mcmc(**kwargs)
        else:
            self.run_map()
        return self

    def create_initial_parameters(self, n=2):
        for _ in range(n):
            yield {k: np.random.random() for k in self.keys}

    def run_mcmc(self, initial_parameters=None, niter=1e1, ncpus=None,
                 nwalkers=None, pool=None, use_pt=False, ntemps=40,
                 skip_map_estimate=False,
                 **emcee_kwargs):
        """ 
        Setup the MCMC machinery and execute a scan over the hyperparameters 
        """
        if ncpus is None:
            ncpus = multiprocessing.cpu_count()
        if nwalkers is None:
            nwalkers = max(ncpus * 2, len(self.keys) * 2)
        if use_pt is False: 
            ntemps = 1
        if initial_parameters is None:
            if self.initial_parameters is None and use_pt is not None:
                n = ntemps * nwalkers
                initial_parameters = [ip.values() for ip in
                                      self.create_initial_parameters(n=n)]
                initial_parameters = np.array(initial_parameters)
                single_parameter = initial_parameters[0]
                if use_pt:
                    initial_parameters.shape = (ntemps, nwalkers, len(self.keys))
            else:
                initial_parameters = self.initial_parameters

        # Run the MCMC sampler -- ensure the proper close of the threads
        fposterior = Posterior(self.xtrain,
                               self.dtrain,
                               single_parameter,
                               self.func_hyperconstraint,
                               self.keys,
                               self.model,
                               self.func_mean,
                               self.func_covariance,
                               self.func_error,
                               self.fixed_mean,
                               self.fixed_covariance)

        # Print to debugging the initial params
        msg  = "Using initial parameters for walker #1: \n"
        msg += str(self.devectorize_args(single_parameter)) + "\n"
        self.logger.debug(msg)

        pickle.dumps(fprior)
        pickle.dumps(fposterior)
        if ncpus > 1 or pool is not None:
            if pool is None:
                pool = multiprocessing.Pool(processes=ncpus)
            if use_pt == False:
                mcs = emcee.EnsembleSampler(nwalkers, len(self.keys),
                                            fposterior, pool=pool,
                                            **emcee_kwargs)
            else:
                # Use a flat prior
                mcs = emcee.PTSampler(ntemps, nwalkers, len(self.keys),
                                      fprior, fposterior)
        else:
            if use_pt == False:
                mcs = emcee.EnsembleSampler(nwalkers, len(self.keys),
                                            fposterior,
                                            **emcee_kwargs)
            else:
                mcs = emcee.PTSampler(ntemps, nwalkers, len(self.keys),
                                      fprior, fposterior, pool=pool)
        t0 = time.time()
        mcs.run_mcmc(initial_parameters, niter)
        t1 = time.time()
        niter = mcs.chain.shape[0] * mcs.chain.shape[1]
        time_total = t1 - t0
        time_loop = time_total * 1000.0 / niter
        msg = "MCMC finished in %1.1fs, %1.1fms per loop"
        self.logger.debug(msg % (time_total, time_loop))
        print(msg % (time_total, time_loop))
        self.mcmc_sampler = mcs
        if pool:
            pool.close()
            del pool

        # Evaluate the MAP estimate from MCMC chain
        idx = np.where(mcs.lnprobability == mcs.lnprobability.max())
        mle = mcs.chain[idx][0]
        self.mcmc_estimate = self.devectorize_args(mle)
        msg  = "MAP Estimate from MCMC: \n"
        msg += str(self.mcmc_estimate) + "\n"
        self.logger.debug(msg)
        if not skip_map_estimate:
            # Further optimize the MAP estimate
            self.map_estimate = self.run_map(initial_parameters=mle,
                                             niter=1)

        # Find the marginal
        marginal_estimate = {}
        chain = self.mcmc_sampler.chain
        half = chain.shape[1] / 2
        dat = chain[:,half:,:].reshape((-1, len(self.keys)))
        for axis, key in enumerate(self.keys):
            col = dat[:,axis]
            cnt, edges = np.histogram(col, bins=30)
            centers = 0.5 * (edges[1:] + edges[:-1])
            i = cnt.argmax()
            marginal_estimate[key] = centers[i]
        self.marginal_estimate = marginal_estimate
        if skip_map_estimate:
            _, self.model_instance = fposterior(self.marginal_estimate, 
                                                return_model=True)
        else:
            _, self.model_instance = fposterior(self.map_estimate, 
                                                return_model=True)

        self.used_initial_parameters = initial_parameters
        return self.map_estimate

    def run_map(self, initial_parameters=None,  niter=100, stepsize=1.0,
                T=1000, **kwargs):
        """
        Run a basin-hopping algorithm to avoid local maxima,
        and in each basin run Nelder-Mead to find the maximumum.
        Basinhopping is great when there's lots of minima
        and explores a lot of space while being better than 
        brute force grid searching. Nelder-Mead is ideal when 
        we can't directly access the derivative. 
        """
        def printall(*x, **kw):
            self.logger.debug("Basinhopping: %s %s", x, kw)
            print("Basinhopping: %s %s", x, kw)

        if initial_parameters is None:
            if self.initial_parameters is None:
                initial_parameters = self.create_initial_parameters(n=1)
                initial_parameters = initial_parameters.next().values()
                initial_parameters = np.array(initial_parameters)
            else:
                initial_parameters = self.initial_parameters
        min_kw = {'method': 'nelder-mead', 'tol': 0.5}

        # Run the MCMC sampler -- ensure the proper close of the threads
        fposterior = Posterior(self.xtrain,
                               self.dtrain,
                               initial_parameters,
                               self.func_hyperconstraint,
                               list(self.keys),
                               self.model,
                               self.func_mean,
                               self.func_covariance,
                               self.func_error,
                               self.fixed_mean,
                               self.fixed_covariance,
                               negate=True)

        # Calculate & log pre-optimization params & logp
        self.log_posterior, self.model_instance = \
                fposterior(initial_parameters, return_model=True)
        self.log_posterior *= -1.0  # negate the log_posterior 
        self.used_initial_parameters = initial_parameters
        msg  = "MAP Estimate before optimization: \n"
        msg += str(self.devectorize_args(initial_parameters)) + "\n"
        msg += "logp is %1.2f" % self.log_posterior
        self.logger.debug(msg)

        # Use basinhopping to find MAP
        if niter > 1:
            ret = basinhopping(fposterior, initial_parameters,
                               niter_success=niter, stepsize=stepsize, T=T,
                               minimizer_kwargs=min_kw, callback=printall,
                               **kwargs)
            minimum = ret['x']
        else:
            ret = fmin(fposterior, initial_parameters, disp=True, 
                       full_output=True, maxiter=50)
            minimum = ret[0]
        self.map_estimate = self.devectorize_args(minimum)

        # Calculate & print post-optimization params & logp
        self.log_posterior, self.model_instance = \
                fposterior(self.map_estimate, return_model=True)
        self.log_posterior *= -1.0  # negate the log_posterior 
        self.used_initial_parameters = initial_parameters
        msg  = "MAP Estimate after optimization: \n"
        msg += str(self.map_estimate) + "\n"
        msg += "logp is %1.2f" % self.log_posterior
        self.logger.debug(msg)
        return self.map_estimate

class LogUnformPrior:
    def __init__(self):
        pass

    def __call__(self, *args):
        return -np.log(args)

class Posterior:
    """
    Use a 'function object' to wrap the posterior function in a pickleable
    way.
    """
    model_instance = None

    def __init__(self, xtrain, ytrain, hyperparameters, func_hyperconstraint,
                 keys, model, func_mean, func_covariance, func_error,
                 fixed_mean, fixed_covariance, negate=False):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.func_hyperconstraint = func_hyperconstraint
        self.keys = list(keys)
        if type(hyperparameters) is not dict:
            hyperparameters = self.devectorize(hyperparameters)
        self.hyperparameters = hyperparameters
        self.model = model
        self.func_mean = func_mean
        self.func_covariance = func_covariance
        self.func_error = func_error
        self.fixed_mean = fixed_mean
        self.fixed_covariance = fixed_covariance
        self.fm_cache = func_mean(xtrain, **hyperparameters)
        self.fc_cache = func_covariance(xtrain, ytrain, **hyperparameters)
        self.ncalls = 0
        self.negate = negate

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        return_model = kwargs.pop('return_model', False)
        if self.negate:
            logp = -1.0 * self._posterior(*args, **kwargs)
        else:
            logp = self._posterior(*args, **kwargs)
        if return_model:
            return logp, self.model_instance
        return logp

    def devectorize(self, hp):
        assert len(hp) == len(self.keys)
        return  {k: v for k, v in zip(self.keys, hp)}

    def _posterior(self, hyperparameters):
        if type(hyperparameters) is not dict:
            hyperparameters = self.devectorize(hyperparameters)
        if not self.fixed_mean:
            fm = self.func_mean
        else:
            fm = lambda *_, **__: self.fm_cache
        if not self.fixed_covariance:
            fc = self.func_covariance
        else:
            fc = lambda *_, **__: self.fc_cache
        t0 = time.time()
        fe = self.func_error
        gp = self.model(self.xtrain, self.ytrain, fm, fc, fe, hyperparameters,
                        self.func_hyperconstraint)
        self.model_instance = gp
        logp = gp.log_posterior()
        #print logp, np.array(hyperparameters.values())
        t1 = time.time()
        msg = "Posterior._posterior %1.1fms %02i %05i" % ((t1 - t0) * 1000.0, 
            self.ncalls, time.time() - start)
        logging.debug(msg)
        return logp

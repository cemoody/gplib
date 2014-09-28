import numpy as np
import datetime
import toolz  # Actually builds the Python kitchen sink
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
import scipy.stats
from numba.decorators import jit, autojit


PERTURB = 1e-4


class constrain_positive(object):
    def __call__(self, *args):
        return ~np.any(args < 0)


class DoWMean():
    keys = []

    def __init__(self, x, y, order=None):
        """
        Function class that builds a linear regression for every day of week.
        Initialize by providing a training data set X, Y; later calls will
        extrapolate the linear regression to new X

        Example
        -------
        fm = kernel.FuncMean(X, Y)
        x_new = np.arange(100)
        y_new = fm(x_new)
        """
        self.order = order
        self.period = 7
        self.dow_func_params = []
        for dow in range(self.period):
            xi, yi = x[x % self.period == dow], y[x % self.period == dow]
            if order is None:
                slope, intercept, r, p, stderr = scipy.stats.linregress(xi, yi)
                self.dow_func_params.append((slope, intercept))
            else:
                pi = np.poly1d(np.polyfit(xi, yi, order))
                pi(xi)
                self.dow_func_params.append(pi)

    def __call__(self, x, **trash):
        y = np.zeros(x.shape)
        for dow, pi in enumerate(self.dow_func_params):
	    if self.order is None:
		slope, intercept = pi
                idx = (x % self.period == dow)
                y[idx] = slope * x[idx] + intercept
            else:
                idx = x % 7 == dow
                y[idx] = pi(x[idx])
        return y

class DoWPoly():
    keys = []

    def __init__(self, x, y, order):
        """
        Function class that builds a polynomial regression for every day of week.
        Initialize by providing a training data set X, Y; later calls will
        extrapolate the polynomial regression to new X

        Example
        -------
        fm = kernel.FuncMean(X, Y)
        x_new = np.arange(100)
        y_new = fm(x_new)
        """
        self.order = order
        self.period = 7
        self.dow_funcs = []
        for dow in range(self.period):
            xi, yi = x[x % self.period == dow], y[x % self.period == dow]
            pi = np.poly1d(np.polyfit(xi, yi, order))
            self.dow_funcs.append(pi)

    def __call__(self, x, **trash):
        y = np.zeros(x.shape)
        for dow, pi in enumerate(self.dow_funcs):
            idx = (x % self.period == dow)
            y[idx] = pi(x[idx])
        return y


class DoWCovariance():
    keys = ['dayofyear_scale365_amp',
            'dayofyear_amp',
            'dayofweek_scale1_amp',
            'dayofweek_scale7_amp',
            'dayofweek_scale28_amp',
            'dayofweek_scale91_amp',
            'holiday_amp']

    #keys = ['dayofyear_amp',
    #        'dayofweek_scale7_amp',
    #        'holiday_amp']

    keys = ['dayofyear_scale365_amp',
            'dayofyear_amp',
            'dayofweek_scale7_amp',
            'dayofweek_scale28_amp',
            'dayofweek_scale91_amp',
            'holiday_amp']



    def __init__(self, holiday_start_date=None, no_self_covariance=False, 
                 no_dayofyear=False):
        """
        Function class that initializes the day-of-week covariance function.
        One of the terms is the holiday-to-holiday covariance, which requires
        that the date of the first x datapoint be given.

        no_self_covariance skips updating entries along the diagonal. This
        is effectively the error term, which allows us to fit an uncertainty
        to every day. If used in conjunction with an error kernel, set 
        no_self_covariance=True.

        Example
        -------
        start_date = datetime.date(2012, 10, 10)
        fc = kernel.FuncCovariance(holiday_start_date=start_date)
        fc(np.arange(600), np.arange(300))
        """
        self.holiday_start_date = holiday_start_date
        self.no_self_covariance = no_self_covariance
        self.no_dayofyear = no_dayofyear

    def __call__(self, x, y, **kwargs):
        return self.func_covariance(x, y, **kwargs)

    def func_covariance(self, x, y,
                        dayofyear_scale365_amp=1.0,
                        dayofyear_amp = 1.0, 
                        dayofweek_scale1_amp=0.0,
                        dayofweek_scale7_amp=1.0,
                        dayofweek_scale28_amp=1.0,
                        dayofweek_scale91_amp=1.0,
                        holiday_amp=1.0, k=None,
                        **trash):
        # Day of week, 2-day, 7-day, 91-day periodicity, and day of year
        if k is None:
            k = np.zeros([x.shape[0], y.shape[0]])
        self.dow_kernel(x, y, k, dayofyear_scale365_amp, dayofyear_amp, 
                        dayofweek_scale1_amp,
                        dayofweek_scale7_amp, dayofweek_scale28_amp, 
                        dayofweek_scale91_amp, self.no_dayofyear)

        # Holiday covariances
        holidays = self.calc_holidays(start_date=self.holiday_start_date)
        self.holiday_kernel(k, holidays, holiday_amp)

        # Reduce anything along the diagonal.
        if self.no_self_covariance:
            idx = np.eye(k.shape[0], dtype='bool')
            k[idx] *= np.random.random(k.shape[0]) * 1e-6
            #k[idx] += np.random.random(k.shape[0]) * 1e-4
            #k *= (~np.eye(k.shape[0], dtype='bool') + 1e-9)

        # Noise for numerical stability
        # Avoid singular matrices by adding noise along the diagonal
        noise = np.random.random(x.shape[0]) * PERTURB
        self.numerical_noise_kernel(k, noise)
        return np.matrix(k)

    @staticmethod
    @jit('void(f8[:,:], i8[:], f8)', nopython=True)
    def holiday_kernel(k, holidays, holiday_amp):
        nrow = k.shape[0]
        ncol = k.shape[1]
        nholiday = len(holidays)
        for i in range(nholiday):
            for j in range(nholiday):
                row = holidays[i]
                col = holidays[j]
                if col < ncol and row < nrow:
                    k[row, col] += holiday_amp

    @staticmethod
    @toolz.memoize
    def calc_holidays(start_date=datetime.datetime(2012, 10, 1),
                      end_date=datetime.datetime.now()):
        cal = USFederalHolidayCalendar()
        holidays = np.ones(cal.holidays().shape[0])
        holidays = pd.Series(holidays, index=cal.holidays()).resample("D").fillna(0.0)
        h = holidays[start_date:end_date].values
        h = np.where(h)[0] - 1
        return h

    @staticmethod
    @jit('void(f8[:,:],f8[:])', nopython=True)
    def numerical_noise_kernel(k, noise):
        for row in xrange(k.shape[0]):
            k[row, row] += noise[row]

    @staticmethod
    @jit(nopython=True)
    def dow_kernel(x, y, k, dayofyear_scale365_amp, dayofyear_amp, 
                   dayofweek_scale1_amp,
                   dayofweek_scale7_amp, dayofweek_scale28_amp, 
                   dayofweek_scale91_amp, no_dayofyear):
        nrow, ncol = x.shape[0], y.shape[0]
        s7 = 7.**2.0
        s28 = 28.**2.0
        s91 = 91.**2.0
        s365 = 365.**2.0
        for row in xrange(nrow):
            for col in xrange(row % 7, ncol, 7):
                d2 = (row - col)**2.0
                tmp = (dayofweek_scale1_amp**2.0 * np.exp(-d2) +
                       dayofweek_scale7_amp**2.0 * np.exp(-d2 / s7 ) +
                       dayofweek_scale28_amp**2.0 * np.exp(-d2 / s28 ) +
                       dayofweek_scale91_amp**2.0 * np.exp(-d2 / s91))
                if col < ncol:
                    k[row, col] += tmp
        if no_dayofyear:
            return
        for row in xrange(0, nrow):
            for col in xrange(row % 365, ncol, 365):
                d2 = (row - col)**2.0
                #tmp = dayofyear_scale365_amp * np.exp(-d2 / s365)
                tmp = dayofyear_amp**2.0
                k[row, col] += tmp

class DoWError():
    def __init__(self, dow=True):
        self.dow = dow
        if self.dow:
            self.keys = ['e_day%i' % i for i in range(7)]
        else:
            self.keys = ['e_day0']

    def __call__(self, k, e_day0=1.0, e_day1=1.0, e_day2=1.0,
                 e_day3=1.0, e_day4=1.0, e_day5=1.0,
                 e_day6=1.0, **trash):
        # Add gaussian noise around each data point
        noise = np.random.random(k.shape[0]) * PERTURB
        if self.dow:
            self.gaussian_noise_kernel(k, noise, e_day0=e_day0, e_day1=e_day1, e_day2=e_day2,
                                       e_day3=e_day3, e_day4=e_day4, e_day5=e_day5,
                                       e_day6=e_day6)
        else:
            self.gaussian_noise_kernel(k, noise, e_day0=e_day0, e_day1=e_day0, e_day2=e_day0,
                                       e_day3=e_day0, e_day4=e_day0, e_day5=e_day0,
                                       e_day6=e_day0)
        return k

    @staticmethod
    def gaussian_noise_kernel(k, noise, e_day0=1.0, e_day1=1.0, e_day2=1.0,
                              e_day3=1.0, e_day4=1.0, e_day5=1.0,
                              e_day6=1.0):
        j = 0
        for i in range(0, k.shape[0], 7):
            k[i, i] += e_day0**2.0 + noise[j]
            j+=1
        for i in range(1, k.shape[0], 7):
            k[i, i] += e_day1**2.0 + noise[j]
            j+=1
        for i in range(2, k.shape[0], 7):
            k[i, i] += e_day2**2.0 + noise[j]
            j+=1
        for i in range(3, k.shape[0], 7):
            k[i, i] += e_day3**2.0 + noise[j]
            j+=1
        for i in range(4, k.shape[0], 7):
            k[i, i] += e_day4**2.0 + noise[j]
            j+=1
        for i in range(5, k.shape[0], 7):
            k[i, i] += e_day5**2.0 + noise[j]
            j+=1
        for i in range(6, k.shape[0], 7):
            k[i, i] += e_day6**2.0 + noise[j]
            j+=1


from abc import abstractmethod, ABC
import inspect

from iminuit import Minuit
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from ctapipe.reco.reco_algorithms import Reconstructor


class LHFitter(ABC):

    def __init__(self, data, start_parameters=None, bound_parameters=None):

        self.data = data
        self.end_parameters = None
        self.start_parameters = start_parameters
        self.bound_parameters = bound_parameters
        self.names_parameters = list(inspect.signature(self.log_pdf).parameters)
        # self.start_parameters = self.initialize_fit()
        # self.bound_parameters = self.compute_bounds()
        self.error_parameters = None
        self.correlation_matrix = None

    def __str__(self):

        str = 'Start parameters :\n\t{}\n'.format(self.start_parameters)
        str += 'Bound parameters :\n\t{}\n'.format(self.bound_parameters)
        str += 'End parameters :\n\t{}\n'.format(self.end_parameters)
        str += 'Error parameters :\n\t{}\n'.format(self.error_parameters)
        str += 'Log-Likelihood :\t{}'.format(self.log_likelihood(**self.end_parameters))

        return str

    def fit(self, verbose=True, minuit=False, **kwargs):

        if minuit:

            fixed_params = {}
            bounds_params = {}
            start_params = self.start_parameters

            for key, val in self.bound_parameters.items():

                bounds_params['limit_'+ key] = val

            for key in self.names_parameters:

                if key in kwargs.keys():

                    fixed_params['fix_' + key] = True

                else:

                    fixed_params['fix_' + key] = False

            # print(fixed_params, bounds_params, start_params)
            options = {**start_params, **bounds_params, **fixed_params}
            f = lambda *args: -self.log_likelihood(*args)
            print_level = 2 if verbose else 0
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad()
            self.end_parameters = dict(m.values)
            options = {**self.end_parameters, **fixed_params}
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad()
            try:
                self.error_parameters = dict(m.errors)

            except (KeyError, AttributeError, RuntimeError):

                self.error_parameters = {key: np.nan for key in self.names_parameters}
                pass
            # print(self.end_parameters, self.error_parameters)

        else:

            fixed_params = {}

            for param in self.names_parameters:
                if param in kwargs.keys():
                    fixed_params[param] = kwargs[param]
                    del kwargs[param]

            start_parameters = []
            bounds = []
            name_parameters = []

            for key in self.names_parameters:

                if key not in fixed_params.keys():

                    start_parameters.append(self.start_parameters[key])
                    bounds.append(self.bound_parameters[key])
                    name_parameters.append(key)

            def llh(x):

                params = dict(zip(name_parameters, x))
                return -self.log_likelihood(**params, **fixed_params)

            result = minimize(llh, x0=start_parameters, bounds=bounds, **kwargs)
            self.end_parameters = dict(zip(name_parameters, result.x))
            self.end_parameters.update(fixed_params)

            try:
                self.correlation_matrix = result.hess_inv.todense()
                self.error_parameters = dict(zip(name_parameters,
                                             np.diagonal(np.sqrt(self.correlation_matrix))))
            except (KeyError, AttributeError):
                pass

            if verbose:

                print(result)

    def pdf(self, *args, **kwargs):

        return np.exp(self.log_pdf(*args, **kwargs))

    @abstractmethod
    def plot(self):

        pass

    @abstractmethod
    def compute_bounds(self):

        pass

    @abstractmethod
    def log_pdf(self, *args, **kwargs):

        pass

    def likelihood(self, *args, **kwargs):

        return np.exp(self.log_likelihood(*args, **kwargs))

    def log_likelihood(self, *args, **kwargs):

        llh = self.log_pdf(*args, **kwargs)
        return np.sum(llh)

    def plot_1dlikelihood(self, parameter_name, axes=None, size=1000,
                        x_label=None, invert=False):

        key = parameter_name

        if key not in self.names_parameters:

            raise NameError('Parameter : {} not in existing parameters :'
                            '{}'.format(key, self.names_parameters))

        x = np.linspace(self.bound_parameters[key][0], self.bound_parameters[key][1], num=size)
        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        params = copy(self.end_parameters)
        llh = np.zeros(x.shape)

        for i, xx in enumerate(x):

            params[key] = xx
            llh[i] = self.log_likelihood(**params)

        x_label = self.labels[key] if x_label is None else x_label

        if not invert:
            axes.plot(x, -llh, color='r')
            axes.axvline(self.end_parameters[key], linestyle='--', color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axvline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.axvspan(self.bound_parameters[key][0],
                         self.bound_parameters[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
            axes.set_ylabel('-$\ln \mathcal{L}$')
            axes.set_xlabel(x_label)

        else:

            axes.plot(-llh, x, color='r')
            axes.axhline(self.end_parameters[key], linestyle='--',
                         color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axhline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.axhspan(self.bound_parameters[key][0],
                         self.bound_parameters[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
            axes.set_xlabel('-$\ln \mathcal{L}$')
            axes.set_ylabel(x_label)
            axes.xaxis.set_label_position('top')
            # axes.xaxis.tick_top()

        axes.legend(loc='best')
        return axes

    def plot_2dlikelihood(self, parameter_1, parameter_2=None, size=100,
                          x_label=None, y_label=None):

        if isinstance(size, int):
            size = (size, size)

        key_x = parameter_1
        key_y = parameter_2
        x = np.linspace(self.bound_parameters[key_x], self.bound_parameters[key_x], num=size[0])
        y = np.linspace(self.bound_parameters[key_y], self.bound_parameters[key_y], num=size[1])
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        params = copy(self.end_parameters)
        llh = np.zeros(size)

        for i, xx in enumerate(x):
            params[key_x] = xx
            for j, yy in enumerate(y):

                params[key_y] = yy
                llh[i, j] = self.log_likelihood(**params)


        fig = plt.figure()
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_center = [left, bottom, width, height]
        rect_x = [left, bottom + height + spacing, width, 0.2]
        rect_y = [left + width + spacing, bottom, 0.2, height]
        axes = fig.add_axes(rect_center)
        axes_x = fig.add_axes(rect_x)
        axes_y = fig.add_axes(rect_y)
        axes.tick_params(direction='in', top=True, right=True)
        self.plot_1dlikelihood(parameter_name=parameter_1, axes=axes_x)
        self.plot_1dlikelihood(parameter_name=parameter_2, axes=axes_y,
                               invert=True)
        axes_x.tick_params(direction='in', labelbottom=False)
        axes_y.tick_params(direction='in', labelleft=False)

        axes_x.set_xlabel('')
        axes_y.set_ylabel('')
        axes_x.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
        axes_y.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

        x_label = self.labels[key_x] if x_label is None else x_label
        y_label = self.labels[key_y] if y_label is None else y_label

        im = axes.imshow(-llh.T,  origin='lower', extent=[x.min() - dx/2.,
                                                     x.max() - dx/2,
                                                     y.min() - dy/2,
                                                     y.max() - dy/2], aspect='auto')

        axes.scatter(self.end_parameters[key_x], self.end_parameters[key_y],
                     marker='x', color='w', label='Maximum Likelihood')
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(loc='best')
        plt.colorbar(mappable=im, ax=axes_y, label='-$\ln \mathcal{L}$')
        axes_x.set_xlim(x.min(), x.max())
        axes_y.set_ylim(y.min(), y.max())

        return axes

    def plot_likelihood(self, parameter_1, parameter_2=None,
                        axes=None, size=1000,
                          x_label=None, y_label=None):

        if parameter_2 is None:

            return self.plot_1dlikelihood(parameter_name=parameter_1,
                                          axes=axes, x_label=x_label, size=size)

        else:

            return self.plot_2dlikelihood(parameter_1,
                                          parameter_2=parameter_2,
                                          size=size,
                                          x_label=x_label,
                                          y_label=y_label)

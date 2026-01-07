"""Causal inference methods: DiD and Synthetic Control."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from datetime import timedelta


class DifferenceInDifferences:
    """DiD estimator with Newey-West HAC standard errors."""

    def __init__(self, data, outcome='returns', treated_col='treated',
                 post_col='post', treated_post_col='treated_post'):
        self.data = data.copy()
        self.outcome = outcome
        self.treated_col = treated_col
        self.post_col = post_col
        self.treated_post_col = treated_post_col
        self.model = None
        self.treatment_effect = None

    def estimate(self):
        """Estimate treatment effect using DiD regression."""
        formula = f"{self.outcome} ~ {self.treated_col} + {self.post_col} + {self.treated_post_col}"
        self.model = smf.ols(formula=formula, data=self.data).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5}
        )

        self.treatment_effect = self.model.params[self.treated_post_col]
        std_error = self.model.bse[self.treated_post_col]
        p_value = self.model.pvalues[self.treated_post_col]

        return {
            'method': 'Difference-in-Differences',
            'treatment_effect': self.treatment_effect,
            'std_error': std_error,
            'p_value': p_value,
            't_stat': self.model.tvalues[self.treated_post_col],
            'conf_int': self.model.conf_int().loc[self.treated_post_col].tolist(),
            'r_squared': self.model.rsquared,
            'n_obs': int(self.model.nobs)
        }

    def test_parallel_trends(self):
        """Test parallel trends assumption on pre-treatment data."""
        pre_data = self.data[self.data[self.post_col] == 0].copy()

        min_date = pre_data['date'].min()
        pre_data['trend'] = (pre_data['date'] - min_date).dt.days
        pre_data['treated_trend'] = pre_data[self.treated_col] * pre_data['trend']

        formula = f"{self.outcome} ~ {self.treated_col} + trend + treated_trend"
        model = smf.ols(formula=formula, data=pre_data).fit()

        coef = model.params['treated_trend']
        p_value = model.pvalues['treated_trend']

        return {
            'test': 'Parallel Trends',
            'coefficient': coef,
            'p_value': p_value,
            'passes': p_value > 0.05,
            'interpretation': 'Parallel trends assumption holds' if p_value > 0.05 else 'Parallel trends assumption violated'
        }


class SyntheticControl:
    """Synthetic Control Method estimator."""

    def __init__(self, data, treated_unit, outcome='returns',
                 unit_col='ticker', time_col='date', post_col='post'):
        self.data = data.copy()
        self.treated_unit = treated_unit
        self.outcome = outcome
        self.unit_col = unit_col
        self.time_col = time_col
        self.post_col = post_col
        self.weights = None
        self.treatment_effect = None

    def estimate(self):
        """Estimate treatment effect using SCM."""
        treated_data = self.data[self.data[self.unit_col] == self.treated_unit].copy()
        control_data = self.data[self.data[self.unit_col] != self.treated_unit].copy()

        treated_pre = treated_data[treated_data[self.post_col] == 0][self.outcome].values

        control_units = control_data[self.unit_col].unique()
        control_matrix = []

        for unit in control_units:
            unit_data = control_data[
                (control_data[self.unit_col] == unit) &
                (control_data[self.post_col] == 0)
            ][self.outcome].values
            control_matrix.append(unit_data)

        control_matrix = np.array(control_matrix).T

        self.weights = self._optimize_weights(treated_pre, control_matrix)
        synthetic_outcomes = control_matrix @ self.weights
        pre_rmse = np.sqrt(np.mean((treated_pre - synthetic_outcomes) ** 2))

        treated_post = treated_data[treated_data[self.post_col] == 1][self.outcome].values

        control_post_matrix = []
        for unit in control_units:
            unit_data = control_data[
                (control_data[self.unit_col] == unit) &
                (control_data[self.post_col] == 1)
            ][self.outcome].values
            control_post_matrix.append(unit_data)

        control_post_matrix = np.array(control_post_matrix).T
        synthetic_post = control_post_matrix @ self.weights

        self.treatment_effect = np.mean(treated_post - synthetic_post)
        weights_dict = {unit: weight for unit, weight in zip(control_units, self.weights)}

        return {
            'method': 'Synthetic Control',
            'treatment_effect': self.treatment_effect,
            'weights': weights_dict,
            'pre_treatment_rmse': pre_rmse,
            'n_control_units': len(control_units),
            'n_obs_pre': len(treated_pre),
            'n_obs_post': len(treated_post)
        }

    def _optimize_weights(self, treated, controls):
        """Optimize weights to minimize pre-treatment fit."""
        n_controls = controls.shape[1]

        def objective(w):
            synthetic = controls @ w
            return np.sqrt(np.mean((treated - synthetic) ** 2))

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_controls)]
        w0 = np.ones(n_controls) / n_controls

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x


def run_placebo_test(data, method, event_date, treated_unit, n_days_before=30):
    """Run placebo test with fake treatment date."""
    data_copy = data.copy()
    fake_event = event_date - timedelta(days=int(n_days_before))
    data_copy['post_placebo'] = (data_copy['date'] >= fake_event).astype(int)

    if method == 'did':
        data_copy['treated_post_placebo'] = data_copy['treated'] * data_copy['post_placebo']
        formula = "returns ~ treated + post_placebo + treated_post_placebo"
        model = smf.ols(formula=formula, data=data_copy).fit()
        return model.params['treated_post_placebo']

    elif method == 'scm':
        sc = SyntheticControl(data_copy, treated_unit, post_col='post_placebo')
        results = sc.estimate()
        return results['treatment_effect']

    else:
        raise ValueError("Method must be 'did' or 'scm'")

"""
Causal Inference Methods for Equifax Breach Analysis
Implements Difference-in-Differences and Synthetic Control Method
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from datetime import timedelta


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for causal inference.
    """
    
    def __init__(self, data, outcome='returns', treated_col='treated', 
                 post_col='post', treated_post_col='treated_post'):
        """
        Initialize DiD estimator.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with treatment indicators
        outcome : str
            Name of outcome variable
        treated_col : str
            Name of treatment indicator column
        post_col : str
            Name of post-treatment indicator column
        treated_post_col : str
            Name of interaction term column
        """
        self.data = data.copy()
        self.outcome = outcome
        self.treated_col = treated_col
        self.post_col = post_col
        self.treated_post_col = treated_post_col
        self.model = None
        self.treatment_effect = None
        
    def estimate(self):
        """
        Estimate the DiD treatment effect with Newey-West HAC standard errors.
        
        Returns:
        --------
        dict
            Results including treatment effect, std error, p-value
        """
        # Run regression: Y = β0 + β1*treated + β2*post + β3*treated*post + ε
        # Use Newey-West HAC SEs to account for serial correlation and heteroskedasticity
        formula = f"{self.outcome} ~ {self.treated_col} + {self.post_col} + {self.treated_post_col}"
        self.model = smf.ols(formula=formula, data=self.data).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5}
        )
        
        # Extract treatment effect (coefficient on treated_post)
        self.treatment_effect = self.model.params[self.treated_post_col]
        std_error = self.model.bse[self.treated_post_col]
        p_value = self.model.pvalues[self.treated_post_col]
        
        results = {
            'method': 'Difference-in-Differences',
            'treatment_effect': self.treatment_effect,
            'std_error': std_error,
            'p_value': p_value,
            't_stat': self.model.tvalues[self.treated_post_col],
            'conf_int': self.model.conf_int().loc[self.treated_post_col].tolist(),
            'r_squared': self.model.rsquared,
            'n_obs': int(self.model.nobs)
        }
        
        return results
    
    def test_parallel_trends(self):
        """
        Test the parallel trends assumption using pre-treatment data.
        
        Returns:
        --------
        dict
            Test results including coefficient on treated*trend interaction
        """
        # Filter to pre-treatment period
        pre_data = self.data[self.data[self.post_col] == 0].copy()
        
        # Create trend variable (days from earliest date)
        min_date = pre_data['date'].min()
        pre_data['trend'] = (pre_data['date'] - min_date).dt.days
        pre_data['treated_trend'] = pre_data[self.treated_col] * pre_data['trend']
        
        # Run regression: Y = β0 + β1*treated + β2*trend + β3*treated*trend + ε
        formula = f"{self.outcome} ~ {self.treated_col} + trend + treated_trend"
        model = smf.ols(formula=formula, data=pre_data).fit()
        
        # Test if treated*trend coefficient is significantly different from zero
        coef = model.params['treated_trend']
        p_value = model.pvalues['treated_trend']
        
        results = {
            'test': 'Parallel Trends',
            'coefficient': coef,
            'p_value': p_value,
            'passes': p_value > 0.05,  # Null: parallel trends (coef = 0)
            'interpretation': 'Parallel trends assumption holds' if p_value > 0.05 else 'Parallel trends assumption violated'
        }
        
        return results


class SyntheticControl:
    """
    Synthetic Control Method for causal inference.
    """
    
    def __init__(self, data, treated_unit, outcome='returns', 
                 unit_col='ticker', time_col='date', post_col='post'):
        """
        Initialize Synthetic Control estimator.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        treated_unit : str
            Name/ID of the treated unit
        outcome : str
            Name of outcome variable
        unit_col : str
            Name of unit identifier column
        time_col : str
            Name of time column
        post_col : str
            Name of post-treatment indicator
        """
        self.data = data.copy()
        self.treated_unit = treated_unit
        self.outcome = outcome
        self.unit_col = unit_col
        self.time_col = time_col
        self.post_col = post_col
        self.weights = None
        self.treatment_effect = None
        
    def estimate(self):
        """
        Estimate treatment effect using Synthetic Control Method.
        
        Returns:
        --------
        dict
            Results including treatment effect, weights, pre-treatment RMSE
        """
        # Separate treated and control units
        treated_data = self.data[self.data[self.unit_col] == self.treated_unit].copy()
        control_data = self.data[self.data[self.unit_col] != self.treated_unit].copy()
        
        # Get pre-treatment data
        treated_pre = treated_data[treated_data[self.post_col] == 0][self.outcome].values
        
        # Create matrix of control unit outcomes (pre-treatment)
        control_units = control_data[self.unit_col].unique()
        control_matrix = []
        
        for unit in control_units:
            unit_data = control_data[
                (control_data[self.unit_col] == unit) & 
                (control_data[self.post_col] == 0)
            ][self.outcome].values
            control_matrix.append(unit_data)
        
        control_matrix = np.array(control_matrix).T  # Shape: (time, units)
        
        # Optimize weights to minimize pre-treatment RMSE
        self.weights = self._optimize_weights(treated_pre, control_matrix)
        
        # Calculate synthetic control outcomes
        synthetic_outcomes = control_matrix @ self.weights
        
        # Calculate pre-treatment RMSE
        pre_rmse = np.sqrt(np.mean((treated_pre - synthetic_outcomes) ** 2))
        
        # Calculate treatment effect (post-treatment difference)
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
        
        # Treatment effect is average difference in post-period
        self.treatment_effect = np.mean(treated_post - synthetic_post)
        
        # Create weights dictionary
        weights_dict = {unit: weight for unit, weight in zip(control_units, self.weights)}
        
        results = {
            'method': 'Synthetic Control',
            'treatment_effect': self.treatment_effect,
            'weights': weights_dict,
            'pre_treatment_rmse': pre_rmse,
            'n_control_units': len(control_units),
            'n_obs_pre': len(treated_pre),
            'n_obs_post': len(treated_post)
        }
        
        return results
    
    def _optimize_weights(self, treated, controls):
        """
        Optimize weights to minimize pre-treatment RMSE.
        
        Parameters:
        -----------
        treated : np.array
            Treated unit outcomes (pre-treatment)
        controls : np.array
            Control unit outcomes (pre-treatment), shape (time, units)
        
        Returns:
        --------
        np.array
            Optimal weights
        """
        n_controls = controls.shape[1]
        
        # Objective: minimize RMSE
        def objective(w):
            synthetic = controls @ w
            return np.sqrt(np.mean((treated - synthetic) ** 2))
        
        # Constraints: weights sum to 1, all non-negative
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_controls)]
        
        # Initial guess: equal weights
        w0 = np.ones(n_controls) / n_controls
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x


def run_placebo_test(data, method, event_date, treated_unit, n_days_before=30):
    """
    Run a single placebo test by pretending the event happened earlier.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Pre-treatment data only
    method : str
        'did' or 'scm'
    event_date : datetime
        Fake event date for placebo test
    treated_unit : str
        Name of treated unit
    n_days_before : int
        How many days before real event to test
    
    Returns:
    --------
    float
        Estimated treatment effect from placebo test
    """
    # Create fake post indicator
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


if __name__ == "__main__":
    print("Causal inference methods module loaded successfully")

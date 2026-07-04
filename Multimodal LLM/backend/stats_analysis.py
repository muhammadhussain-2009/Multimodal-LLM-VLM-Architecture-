import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StatsAnalysis")

try:
    import numpy as np
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    logger.warning("Scientific computing libraries (numpy, scipy) missing. Using custom pure-Python statistical functions.")
    HAS_SCIPY = False

class StatisticalEvaluator:
    """
    Handles pre-registered statistical calculations: power analysis (Sample Size),
    t-tests for pre/post-test differences, and Cohen's d effect size evaluation.
    """
    
    @staticmethod
    def calculate_required_sample_size(
        power: float = 0.80, 
        alpha: float = 0.05, 
        cohens_d: float = 0.30,
        two_sided: bool = True
    ) -> int:
        """
        Calculates the required sample size per group for a two-sample t-test.
        Approximated formula:
        n = 2 * ((Z_alpha + Z_beta) / d) ^ 2
        """
        # Critical value approximations for normal distribution
        # Z_alpha/2 for alpha=0.05 is 1.96
        if alpha == 0.05:
            z_alpha = 1.95996 if two_sided else 1.64485
        elif alpha == 0.01:
            z_alpha = 2.57583 if two_sided else 2.32635
        else:
            # General fallback approximation
            z_alpha = 1.96
            
        # Z_beta for power=0.80 (beta=0.20) is 0.84162
        if power == 0.80:
            z_beta = 0.84162
        elif power == 0.90:
            z_beta = 1.28155
        elif power == 0.95:
            z_beta = 1.64485
        else:
            z_beta = 0.84162
            
        n = 2 * ((z_alpha + z_beta) / cohens_d) ** 2
        return int(math.ceil(n))

    @staticmethod
    def run_two_sample_t_test(group_control: list, group_treatment: list) -> dict:
        """
        Executes t-test comparison between control (static feedback) and treatment (Socratic feedback).
        Supports SciPy ttest_ind or pure-Python Welch's t-test calculation.
        """
        n1 = len(group_control)
        n2 = len(group_treatment)
        
        if n1 < 2 or n2 < 2:
            return {"error": "Sample size too small for statistical significance."}
            
        # Calculate means
        mean1 = sum(group_control) / n1
        mean2 = sum(group_treatment) / n2
        
        # Calculate variances
        var1 = sum((x - mean1) ** 2 for x in group_control) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group_treatment) / (n2 - 1)
        
        # Welch's t-test statistic
        se = math.sqrt((var1 / n1) + (var2 / n2))
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean2 - mean1) / se
            
        # Degrees of freedom (Welch–Satterthwaite equation)
        num = ((var1 / n1) + (var2 / n2)) ** 2
        den = (((var1 / n1) ** 2) / (n1 - 1)) + (((var2 / n2) ** 2) / (n2 - 1))
        df = num / den if den != 0 else 1.0
        
        # Cohen's d (pooled standard deviation)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        sd_pooled = math.sqrt(pooled_var) if pooled_var > 0 else 1.0
        cohens_d = (mean2 - mean1) / sd_pooled
        
        # Approximate p-value lookup or scipy call
        if HAS_SCIPY:
            t_stat_sp, p_val_sp = stats.ttest_ind(group_treatment, group_control, equal_var=False)
            p_val = p_val_sp
            t_stat = t_stat_sp
        else:
            # Custom rough p-value approximation for Welch's df
            # Z approximation for large df
            z = abs(t_stat)
            # Standard approximation of normal CDF
            p_approx = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
            p_val = p_approx

        return {
            "mean_control": round(mean1, 3),
            "mean_treatment": round(mean2, 3),
            "t_statistic": round(t_stat, 3),
            "degrees_of_freedom": round(df, 1),
            "p_value": round(p_val, 5),
            "cohens_d": round(cohens_d, 3),
            "statistically_significant": p_val < 0.05
        }

if __name__ == "__main__":
    # Test calculations
    n_required = StatisticalEvaluator.calculate_required_sample_size(power=0.80, alpha=0.05, cohens_d=0.30)
    print(f"Required Sample Size per group (80% power, alpha=0.05, d=0.30): {n_required} (Total: {n_required * 2})")
    
    # Mock data representing student test gains:
    # Control group: Traditional direct corrections (mean gain ~12 points)
    # Treatment group: Socratic feedback pipeline (mean gain ~18 points)
    import random
    random.seed(42)
    control_scores = [random.normalvariate(12, 8) for _ in range(n_required)]
    treatment_scores = [random.normalvariate(16.5, 9) for _ in range(n_required)]
    
    results = StatisticalEvaluator.run_two_sample_t_test(control_scores, treatment_scores)
    print("Pre-registered Hypothesis Test Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

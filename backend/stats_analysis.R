# Statistical evaluation script for pre-registered design
# Power target: 80% power, alpha = 0.05, minimum detectable effect size (Cohen's d) = 0.30

# Load packages
if (!requireNamespace("pwr", quietly = TRUE)) {
  install.packages("pwr", repos = "http://cran.us.r-project.org")
}
library(pwr)

# 1. Power Analysis calculation
# Calculate required sample size for a two-tailed independent two-sample t-test
power_analysis <- pwr.t.test(
  d = 0.30,          # Minimum detectable effect size
  sig.level = 0.05,  # Alpha level
  power = 0.80,      # Target power
  type = "two.sample",
  alternative = "two.sided"
)

cat("=========================================\n")
cat("PRE-REGISTERED POWER ANALYSIS SUMMARY\n")
cat("=========================================\n")
print(power_analysis)
cat("Required sample size per group:", ceiling(power_analysis$n), "\n")
cat("Total required sample size across both cohorts:", ceiling(power_analysis$n) * 2, "\n\n")

# 2. Hypothesis Testing Demonstration
# Generating mock experimental data based on expected educational outcomes:
# control_group: Traditional Direct Feedback (mean normalized learning gain = 0.42, sd = 0.15)
# treatment_group: Adaptive Socratic AI Pipeline (mean normalized learning gain = 0.48, sd = 0.15)
set.seed(42)
n_sample <- ceiling(power_analysis$n)

control_group <- rnorm(n_sample, mean = 0.42, sd = 0.15)
treatment_group <- rnorm(n_sample, mean = 0.48, sd = 0.15)

# Run Welch two-sample t-test
t_result <- t.test(treatment_group, control_group, var.equal = FALSE)

# Compute exact Cohen's d
pooled_sd <- sqrt((var(control_group) + var(treatment_group)) / 2)
cohens_d <- (mean(treatment_group) - mean(control_group)) / pooled_sd

cat("=========================================\n")
cat("WELCH'S TWO-SAMPLE T-TEST RESULTS\n")
cat("=========================================\n")
print(t_result)
cat("Estimated Effect Size (Cohen's d):", round(cohens_d, 3), "\n")

if (t_result$p.value < 0.05) {
  cat("Conclusion: The Socratic feedback engine shows a statistically significant improvement in student learning gains (p < 0.05).\n")
} else {
  cat("Conclusion: The difference in learning gains did not reach statistical significance (p >= 0.05).\n")
}
cat("=========================================\n")

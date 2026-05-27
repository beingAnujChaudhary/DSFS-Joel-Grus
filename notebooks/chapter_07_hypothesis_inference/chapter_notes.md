
# Chapter 7 — Hypothesis and Inference

## Chapter Overview

What do we do with all the probability and statistics we've learned? The "science" part of data science frequently involves forming and testing hypotheses about our data and the processes that generate it.

This chapter transitions from descriptive statistics and probability into **statistical inference**. Joel Grus covers the classical frequentist approach (null hypotheses, p-values, confidence intervals, and A/B testing) and contrasts it with the Bayesian approach (treating unknown parameters as random variables and updating beliefs with priors and posteriors).

The chapter also highlights the dangers of **p-hacking**, demonstrating how easily statistical significance can be manipulated or found by pure chance if hypotheses aren't formulated rigorously.

The chapter introduces:
- Statistical inference and hypothesis testing
- Null and alternative hypotheses
- Statistical significance and p-values
- Type I and Type II errors
- Confidence intervals
- P-hacking and its dangers
- A/B testing
- Bayesian inference with conjugate priors

**The key lesson:**

> Data helps us make decisions under uncertainty. Statistics doesn't prove things are true; it quantifies how surprised we should be by the data if our assumptions were true.

**The key mindset:**

> How confident am I that this result is real and not random noise?

---

## Learning Objectives

After completing this chapter, I should be able to:

- Formulate null ($H_0$) and alternative ($H_1$) hypotheses for a given problem
- Calculate the significance (Type I error) and power (1 - Type II error) of a statistical test
- Compute and correctly interpret p-values, including applying continuity corrections
- Construct and interpret confidence intervals for estimated parameters
- Identify and avoid p-hacking in data analysis
- Design and evaluate an A/B test comparing two proportions
- Apply Bayesian inference using the Beta distribution as a conjugate prior for Binomial data
- Explain why hypothesis testing matters in ML and experimentation

---

## Key Terms

| Term | Definition |
|------|------------|
| **Hypothesis** | A claim that can be tested using data |
| **Null Hypothesis ($H_0$)** | The default assumption that nothing has changed |
| **Alternative Hypothesis ($H_1$)** | The claim that something has changed |
| **Statistical Significance** | A result is significant if it's unlikely to occur by random chance |
| **p-value** | Probability of observing a result at least as extreme if $H_0$ were true |
| **Type I Error (False Positive)** | Rejecting a true null hypothesis |
| **Type II Error (False Negative)** | Failing to reject a false null hypothesis |
| **Significance Level ($\alpha$)** | The threshold for rejecting $H_0$ (typically 0.05) |
| **Power ($1 - \beta$)** | Probability of correctly rejecting a false $H_0$ |
| **Confidence Interval** | A range expressing uncertainty around an estimate |
| **P-hacking** | Manipulating data or analysis to find significant results |
| **A/B Test** | Comparing two variants to determine which performs better |
| **Bayesian Inference** | Updating beliefs about parameters using prior knowledge and observed data |
| **Beta Distribution** | A conjugate prior for Binomial data in Bayesian analysis |
| **Conjugate Prior** | A prior distribution that yields a posterior in the same family |

---

## Mental Model: Inference as a Decision-Making Framework

> Statistics doesn't prove things are true; it quantifies how surprised we should be by the data if our assumptions were true.

Think of inference as a structured debate between a default assumption and a new claim:

```txt
Claim → Collect Evidence → Statistical Test → Decision Under Uncertainty
```

Two major philosophical approaches:
- **Frequentist approach:** "Assuming the default is true, how weird is this data?" (p-values, confidence intervals)
- **Bayesian approach:** "Given my prior beliefs and this new data, what is the updated probability of my hypothesis?" (posteriors)

```txt
Good inference = Pre-defined hypotheses + Rigorous testing + Awareness of systemic biases
```

**Key questions to ask:**
- Is this result meaningful, or just random chance?
- How confident am I in this conclusion?
- What errors might I be making?

---

## Key Concepts

### 1. Why Inference Matters

Real-world decisions often rely on limited data:

```
Did a medicine work?
Did a website redesign improve clicks?
Did a new ML model improve accuracy?
Is variant A better than variant B?
```

We rarely have certainty. Instead, **we estimate confidence**.

---

### 2. Statistical Hypothesis Testing

We start with a **null hypothesis ($H_0$)** (the default position, e.g., "the coin is fair," $p = 0.5$) and an **alternative hypothesis ($H_1$)** (e.g., $p \neq 0.5$).

Because we're dealing with Binomial random variables (coin flips), we use the Central Limit Theorem to approximate them with a Normal distribution:

```python
def normal_approximation_to_binomial(n, p):
    """Finds mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
```

#### Significance and Power

- **Significance ($\alpha$):** Probability of Type I error (false positive). We reject $H_0$ even though it's true. Usually set at 5% (0.05).
- **Power ($1 - \beta$):** Probability of correctly rejecting $H_0$ when it's false. The probability of not making a Type II error.

```python
# 95% bounds based on assumption p = 0.5
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)  # (469, 531)

# Actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# Power = 1 - P(Type II error)
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability  # ~0.887
```

---

### 3. p-values and Continuity Correction

Instead of choosing bounds beforehand, a **p-value** is the probability of seeing a value *at least as extreme* as the one we observed, assuming $H_0$ is true.

```python
def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)
```

**Interpreting p-values:**
- **Small p-value (e.g., < 0.05):** Strong evidence against $H_0$
- **Large p-value (e.g., > 0.05):** Weak evidence — data is consistent with $H_0$
- **p-value is NOT proof** — it's a measure of surprise, not a measure of effect size or practical importance

**Continuity Correction:** Because we're approximating a discrete distribution (Binomial) with a continuous one (Normal), we use `529.5` instead of `530` to better estimate the probability of seeing *exactly* 530 heads.

---

### 4. Type I and Type II Errors

| | $H_0$ is TRUE | $H_0$ is FALSE |
|---|---|---|
| **Reject $H_0$** | Type I Error (False Positive) | Correct (True Positive) |
| **Fail to Reject $H_0$** | Correct (True Negative) | Type II Error (False Negative) |

**Examples:**

**Type I Error (False Positive):**
Believing a medicine works when it actually doesn't.

**Type II Error (False Negative):**
Believing a medicine doesn't work when it actually does.

**Trade-off:** Reducing Type I errors increases Type II errors (and vice versa), given a fixed sample size.

---

### 5. Confidence Intervals

Instead of testing a hypothesis about a parameter $p$, we can estimate $p$ from the data and build an interval around it.

```python
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)

# 95% Confidence Interval
normal_two_sided_bounds(0.95, mu, sigma)  # [0.4940, 0.5560]
```

**Crucial Interpretation:** The 95% refers to the *method*, not the specific parameter. If we repeated this experiment many times, 95% of the calculated intervals would contain the true, fixed parameter $p$.

Instead of saying "Accuracy = 90%", think "Accuracy ≈ 90% ± some uncertainty" (e.g., 90% ± 3%).

---

### 6. P-hacking

If a procedure erroneously rejects the null hypothesis 5% of the time, then by definition, testing 1,000 true null hypotheses will yield ~50 "significant" results purely by chance.

**Common p-hacking tactics:**
- Testing many hypotheses and only reporting significant ones
- Stopping data collection once significance is reached
- Tweaking analysis after seeing the data
- Trying different model configurations until one works

**Rules to avoid p-hacking:**
- Determine your hypotheses *before* looking at the data
- Clean your data without the hypotheses in mind
- Use correction methods (Bonferroni, Benjamini-Hochberg) for multiple comparisons
- p-values are not a substitute for common sense

---

### 7. A/B Testing

To compare two ads (A and B), we treat the click-through rates as two independent normal random variables. We test the null hypothesis that $p_A - p_B = 0$.

```python
def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
```

If the resulting z-score yields a p-value < 0.05, we conclude there is a statistically significant difference between the ads.

**A/B testing in ML:** Comparing recommendation systems, ranking algorithms, model variants, and user interfaces.

---

### 8. Bayesian Inference

Instead of making probability statements about the *test*, Bayesian inference makes probability statements about the *parameters themselves*.

We start with a **prior distribution** and update it with observed data using Bayes's Theorem to get a **posterior distribution**.

For a Binomial likelihood (coin flips), the **Beta distribution** is a *conjugate prior*. This means if the prior is Beta($\alpha, \beta$), the posterior after observing $h$ heads and $t$ tails is simply Beta($\alpha + h, \beta + t$).

```python
def B(alpha, beta):
    """Normalizing constant so total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1: return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
```

**The Beauty of Conjugate Priors:**
- Start with a uniform prior: Beta(1, 1). See 3 heads, 7 tails → Posterior: Beta(4, 8).
- Start with a strong belief it's fair: Beta(20, 20). See 3 heads, 7 tails → Posterior: Beta(23, 27).

As data accumulates, the prior's influence washes out, and the posteriors converge toward the observed frequency.

---

## Important Takeaways

### Statistical significance ≠ Practical significance
A tiny difference can be "statistically significant" if your sample size is massive, but it might not matter for business logic. Always consider effect size alongside p-values.

### Confidence intervals are about the process
Saying "there is a 95% probability the true parameter is in this specific interval" is technically incorrect in frequentist statistics. The parameter is fixed; the interval is random.

### P-hacking is a systemic flaw
Testing multiple hypotheses without correction guarantees false discoveries. Modern solutions include the Bonferroni correction or False Discovery Rate control.

### Bayesian inference allows intuitive statements
Bayesian methods let you say "there is a 95% probability the coin's bias is between 0.4 and 0.6," which is what most people intuitively want to know. This is philosophically different from frequentist confidence intervals.

### Data never gives perfect certainty
Instead of "proof," we have evidence, confidence, and probability. Decisions must be made under uncertainty.

### Small p-value ≠ truth
It only means the result is less likely under $H_0$. It doesn't measure the probability that $H_0$ is true, nor does it measure effect size.

---

## Connections to AI/ML Engineering

This chapter is foundational for production ML and experimentation:

| Inference Concept | ML/Data Science Application |
|-------------------|-----------------------------|
| A/B Testing | Evaluating new ML models in production (shadow deployment, canary releases, online experiments) |
| Confidence Intervals | Reporting model metrics (e.g., "Accuracy is 92% ± 2%") to stakeholders |
| P-hacking awareness | Preventing overfitting to the validation set during hyperparameter tuning |
| Type I/II Error analysis | Setting thresholds for fraud detection, medical diagnosis, content moderation |
| Bayesian Inference | Bayesian Optimization for hyperparameter tuning, Thompson Sampling in Multi-Armed Bandits (RL) |
| Conjugate Priors | Building probabilistic graphical models and updating beliefs in streaming data environments |

### Specific Applications:

**A/B Testing:**
- Comparing recommendation systems
- Evaluating ranking algorithms
- Testing UI/UX changes

**Model Evaluation:**
- Is the new model *actually* better?
- Quantifying improvement with confidence intervals

**Experiment Tracking:**
- Understanding real improvement vs. random variation
- Making go/no-go decisions on model deployments

**Production Monitoring:**
- Determining signal vs. noise in metrics
- Detecting meaningful drift vs. random fluctuation

**Production AI Systems:**
- Many business decisions rely on statistical confidence
- Model deployment decisions require rigorous validation

---

## Questions / Confusions

Things I still want to understand better:

- [ ] Why does the p-value work mathematically?
- [ ] How do I correct for multiple comparisons (Bonferroni, Benjamini-Hochberg) when testing hundreds of ML features?
- [ ] When should I use a t-distribution instead of a normal approximation for A/B tests?
- [ ] How do Markov Chain Monte Carlo (MCMC) methods work for Bayesian inference when conjugate priors aren't available?
- [ ] How does the Bayesian approach handle high-dimensional priors in deep learning (Bayesian Neural Networks)?
- [ ] How is inference used in production ML monitoring and alerting?
- [ ] When should statistical significance matter most in business decisions?

---

## Personal Reflections

### What surprised me?
The section on p-hacking is terrifying. It's shockingly easy to "find" a significant result just by running enough random experiments or tweaking data cleaning steps after looking at the data. This completely changes how I view "statistically significant" claims in research papers and business reports.

### What stood out?
The elegance of the Beta-Binomial conjugate prior. The fact that you can just add observed heads to $\alpha$ and tails to $\beta$ to update your entire belief distribution is mathematically beautiful and computationally trivial. Also: small model improvements may be meaningless without significance testing — a 0.1% accuracy gain might just be noise.

### How does this connect to my roadmap?
In production ML, the job isn't done when the model is trained. You have to prove it works better than the baseline. A/B testing and confidence intervals are the exact tools used to make go/no-go decisions on model deployments. Furthermore, Bayesian concepts are the foundation of modern AutoML and hyperparameter optimization tools like Optuna. This chapter is essential for ML evaluation, experimentation, A/B testing, production decision-making, and model comparison. Strong statistical reasoning will improve ML judgment significantly.

---

## Chapter Summary (In My Own Words)

Chapter 7 bridges the gap between probability theory and real-world decision-making. It introduces the frequentist toolkit — hypothesis testing, p-values, and confidence intervals — while heavily warning against the misuse of these tools through p-hacking. It then demonstrates how to apply these concepts to A/B testing, the bread-and-butter of tech industry experimentation. Finally, it introduces Bayesian inference, shifting the paradigm from "what is the probability of the data given the hypothesis" to "what is the probability of the hypothesis given the data," utilizing the Beta distribution as a powerful conjugate prior.

The core lessons are:

1. **Hypothesis testing provides a framework** for making decisions under uncertainty
2. **Null hypothesis is the default** — start with "nothing changed" and test against evidence
3. **p-values measure surprise**, not truth or effect size
4. **Errors are inevitable** — understanding Type I and Type II trade-offs matters
5. **Confidence intervals express uncertainty** — more informative than point estimates
6. **P-hacking is dangerous** — pre-register hypotheses and correct for multiple comparisons
7. **Bayesian inference allows intuitive probability statements** about parameters

The ultimate lesson is that statistical inference is a rigorous framework for managing uncertainty, but it requires discipline, pre-defined hypotheses, and a deep understanding of its underlying assumptions to avoid fooling oneself.

> "It is the mark of a truly intelligent person to be moved by statistics." — George Bernard Shaw

---

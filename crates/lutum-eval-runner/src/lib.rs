use std::{fmt, future::Future};

use futures::{StreamExt as _, stream};

/// A single k-metric value.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct KMetric {
    pub k: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KMetricRangeError {
    pub k: u64,
    pub n: u64,
}

impl fmt::Display for KMetricRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "k must be <= n (got k={}, n={})", self.k, self.n)
    }
}

impl std::error::Error for KMetricRangeError {}

/// Estimated probability that at least one of `k` randomly sampled trials
/// passes, given `c` passes out of `n` total trials.
///
/// Uses the unbiased estimator from Chen et al. (2021):
/// `1 - C(n-c, k) / C(n, k)`
pub fn pass_at_k(n: u64, c: u64, k: u64) -> f64 {
    assert!(c <= n, "c must be <= n");
    assert!(k <= n, "k must be <= n");

    if k == 0 {
        return 0.0;
    }
    if c == 0 {
        return 0.0;
    }
    if n - c < k {
        return 1.0;
    }

    let mut fail_prob = 1.0f64;
    for j in 0..k {
        fail_prob *= (n - c - j) as f64 / (n - j) as f64;
    }
    1.0 - fail_prob
}

/// Estimated probability that all `k` randomly sampled trials pass, given `c`
/// passes out of `n` total trials.
pub fn pass_hat_k(n: u64, c: u64, k: u64) -> f64 {
    assert!(c <= n, "c must be <= n");
    assert!(k <= n, "k must be <= n");

    if k == 0 {
        return 1.0;
    }
    if c < k {
        return 0.0;
    }

    let mut all_pass_prob = 1.0f64;
    for j in 0..k {
        all_pass_prob *= (c - j) as f64 / (n - j) as f64;
    }
    all_pass_prob
}

/// Mean pass@k across multiple tasks, each with their own (n, c) counts.
pub fn mean_pass_at_k(counts: &[(u64, u64)], k: u64) -> f64 {
    assert!(!counts.is_empty(), "counts must not be empty");
    counts.iter().map(|&(n, c)| pass_at_k(n, c, k)).sum::<f64>() / counts.len() as f64
}

/// Mean pass^k across multiple tasks, each with their own (n, c) counts.
pub fn mean_pass_hat_k(counts: &[(u64, u64)], k: u64) -> f64 {
    assert!(!counts.is_empty(), "counts must not be empty");
    counts
        .iter()
        .map(|&(n, c)| pass_hat_k(n, c, k))
        .sum::<f64>()
        / counts.len() as f64
}

/// Compute `KMetric` values for k = 1..=max_k using the given metric function.
///
/// `metric` must be defined for every `k` in `1..=max_k`.
pub fn k_metrics(
    counts: &[(u64, u64)],
    max_k: usize,
    metric: impl Fn(&[(u64, u64)], u64) -> f64,
) -> Vec<KMetric> {
    (1..=max_k)
        .map(|k| KMetric {
            k,
            value: metric(counts, k as u64),
        })
        .collect()
}

/// A collection of trial results with aggregation methods.
///
/// Treats `Ok(_)` as a passing trial. For score-based thresholds, filter
/// results before constructing `TrialSet`.
pub struct TrialSet<T, E> {
    trials: Vec<Result<T, E>>,
}

impl<T, E> TrialSet<T, E> {
    pub fn new(trials: Vec<Result<T, E>>) -> Self {
        Self { trials }
    }

    pub fn len(&self) -> usize {
        self.trials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }

    pub fn pass_count(&self) -> usize {
        self.trials.iter().filter(|r| r.is_ok()).count()
    }

    pub fn fail_count(&self) -> usize {
        self.trials.iter().filter(|r| r.is_err()).count()
    }

    pub fn trials(&self) -> &[Result<T, E>] {
        &self.trials
    }

    pub fn into_trials(self) -> Vec<Result<T, E>> {
        self.trials
    }

    /// (n, c) pair for use with `pass_at_k` / `pass_hat_k` / `k_metrics`.
    pub fn counts(&self) -> (u64, u64) {
        let n = self.trials.len() as u64;
        let c = self.pass_count() as u64;
        (n, c)
    }

    pub fn pass_at_k(&self, k: u64) -> f64 {
        let (n, c) = self.counts();
        pass_at_k(n, c, k)
    }

    pub fn pass_hat_k(&self, k: u64) -> f64 {
        let (n, c) = self.counts();
        pass_hat_k(n, c, k)
    }

    /// Returns pass@k for every valid k in `1..=n`, where `n = self.len()`.
    pub fn k_metrics_pass_at(&self) -> Vec<KMetric> {
        let counts = [self.counts()];
        k_metrics(&counts, self.len(), mean_pass_at_k)
    }

    /// Returns pass^k for every valid k in `1..=n`, where `n = self.len()`.
    pub fn k_metrics_pass_hat(&self) -> Vec<KMetric> {
        let counts = [self.counts()];
        k_metrics(&counts, self.len(), mean_pass_hat_k)
    }

    pub fn k_metrics_pass_at_checked(
        &self,
        max_k: usize,
    ) -> Result<Vec<KMetric>, KMetricRangeError> {
        let (n, _) = self.counts();
        validate_k(n, max_k as u64)?;
        let counts = [self.counts()];
        Ok(k_metrics(&counts, max_k, mean_pass_at_k))
    }

    pub fn k_metrics_pass_hat_checked(
        &self,
        max_k: usize,
    ) -> Result<Vec<KMetric>, KMetricRangeError> {
        let (n, _) = self.counts();
        validate_k(n, max_k as u64)?;
        let counts = [self.counts()];
        Ok(k_metrics(&counts, max_k, mean_pass_hat_k))
    }
}

fn validate_k(n: u64, k: u64) -> Result<(), KMetricRangeError> {
    if k > n {
        return Err(KMetricRangeError { k, n });
    }
    Ok(())
}

/// Runs `n` independent trials with up to `concurrency` in flight at once.
///
/// `f(trial_index)` produces the future for each trial. Results are returned
/// in completion order (not submission order).
pub async fn run_parallel<T, E, F, Fut>(n: usize, concurrency: usize, f: F) -> Vec<Result<T, E>>
where
    T: Send + 'static,
    E: Send + 'static,
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<T, E>> + Send,
{
    stream::iter((0..n).map(f))
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await
}

/// Runs `n` trials and returns a [`TrialSet`] in completion order.
pub async fn run_trials<T, E, F, Fut>(n: usize, concurrency: usize, f: F) -> TrialSet<T, E>
where
    T: Send + 'static,
    E: Send + 'static,
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<T, E>> + Send,
{
    let trials = run_parallel(n, concurrency, f).await;
    TrialSet::new(trials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_at_k1_equals_empirical_success_rate() {
        let n = 5;
        let c = 2;
        assert!((pass_at_k(n, c, 1) - 0.4).abs() < 1e-12);
        assert!((pass_hat_k(n, c, 1) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn pass_at_k_small_known_values() {
        let n = 5;
        let c = 2;
        assert!((pass_at_k(n, c, 2) - 0.7).abs() < 1e-12);
        assert!((pass_hat_k(n, c, 2) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn pass_at_k_boundary_cases() {
        assert_eq!(pass_at_k(10, 0, 3), 0.0);
        assert_eq!(pass_hat_k(10, 0, 3), 0.0);
        assert_eq!(pass_at_k(10, 10, 3), 1.0);
        assert_eq!(pass_hat_k(10, 10, 3), 1.0);
    }

    #[test]
    fn pass_hat_k_insufficient_successes() {
        assert_eq!(pass_hat_k(10, 3, 4), 0.0);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn pass_at_k_panics_when_k_exceeds_n() {
        let _ = pass_at_k(5, 2, 6);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn pass_hat_k_panics_when_k_exceeds_n() {
        let _ = pass_hat_k(5, 2, 6);
    }

    #[test]
    fn mean_metrics_average_per_task() {
        let counts = vec![(5, 2), (5, 5)];
        let expected_pass_at_2 = (0.7 + 1.0) / 2.0;
        let expected_pass_hat_2 = (0.1 + 1.0) / 2.0;
        assert!((mean_pass_at_k(&counts, 2) - expected_pass_at_2).abs() < 1e-12);
        assert!((mean_pass_hat_k(&counts, 2) - expected_pass_hat_2).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn mean_pass_at_k_panics_on_mixed_trial_counts() {
        let counts = vec![(1, 1), (10, 10)];
        let _ = mean_pass_at_k(&counts, 2);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn mean_pass_hat_k_panics_on_mixed_trial_counts() {
        let counts = vec![(1, 1), (10, 10)];
        let _ = mean_pass_hat_k(&counts, 2);
    }

    #[test]
    fn trial_set_counts_and_metrics() {
        let trials: Vec<Result<(), ()>> = vec![Ok(()), Ok(()), Err(()), Ok(()), Err(())];
        let set = TrialSet::new(trials);
        assert_eq!(set.len(), 5);
        assert_eq!(set.pass_count(), 3);
        assert_eq!(set.fail_count(), 2);
        assert_eq!(set.counts(), (5, 3));
        let metrics = set.k_metrics_pass_at();
        assert_eq!(metrics.len(), 5);
    }

    #[test]
    fn trial_set_k_metrics_cover_all_valid_ks() {
        let trials: Vec<Result<(), ()>> = vec![Ok(()), Err(()), Ok(())];
        let set = TrialSet::new(trials);
        let pass_at = set.k_metrics_pass_at();
        let pass_hat = set.k_metrics_pass_hat();

        assert_eq!(pass_at.len(), 3);
        assert_eq!(pass_hat.len(), 3);
        assert_eq!(pass_at[2].k, 3);
        assert_eq!(pass_hat[2].k, 3);
    }

    #[test]
    fn trial_set_checked_metrics_allow_shorter_prefixes() {
        let trials: Vec<Result<(), ()>> = vec![Ok(()), Err(()), Ok(())];
        let set = TrialSet::new(trials);
        let pass_at = set.k_metrics_pass_at_checked(2).unwrap();
        let pass_hat = set.k_metrics_pass_hat_checked(2).unwrap();

        assert_eq!(pass_at.len(), 2);
        assert_eq!(pass_hat.len(), 2);
    }

    #[test]
    fn trial_set_checked_metrics_reject_oversized_k() {
        let trials: Vec<Result<(), ()>> = vec![Ok(()), Err(()), Ok(())];
        let set = TrialSet::new(trials);

        assert_eq!(
            set.k_metrics_pass_at_checked(5),
            Err(KMetricRangeError { k: 5, n: 3 })
        );
        assert_eq!(
            set.k_metrics_pass_hat_checked(5),
            Err(KMetricRangeError { k: 5, n: 3 })
        );
    }

    #[tokio::test]
    async fn run_trials_returns_all_results() {
        let set = run_trials(5, 3, |i| async move {
            if i % 2 == 0 { Ok::<_, ()>(i) } else { Err(()) }
        })
        .await;
        assert_eq!(set.len(), 5);
        assert_eq!(set.pass_count(), 3); // indices 0, 2, 4
    }
}

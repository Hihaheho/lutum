//! Core budget primitives intentionally live in the shared protocol crate.
//!
//! Budget policy is user-driven through the [`BudgetManager`] trait, so
//! adapters and higher-level crates can plug in their own accounting and
//! enforcement strategies without moving budget ownership out of core.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use thiserror::Error;

use crate::{AgentError, RequestExtensions};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UsageEstimate {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_micros_usd: u64,
}

impl UsageEstimate {
    pub const fn zero() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_micros_usd: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_micros_usd: u64,
}

impl Usage {
    pub const fn zero() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_micros_usd: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Remaining {
    pub tokens: u64,
    pub cost_micros_usd: u64,
    pub below_threshold: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequestBudget {
    pub tokens: Option<u64>,
    pub cost_micros_usd: Option<u64>,
}

impl RequestBudget {
    pub const fn unlimited() -> Self {
        Self {
            tokens: None,
            cost_micros_usd: None,
        }
    }

    pub const fn from_tokens(tokens: u64) -> Self {
        Self {
            tokens: Some(tokens),
            cost_micros_usd: None,
        }
    }

    pub const fn from_cost_micros_usd(cost_micros_usd: u64) -> Self {
        Self {
            tokens: None,
            cost_micros_usd: Some(cost_micros_usd),
        }
    }

    pub const fn with_limits(tokens: Option<u64>, cost_micros_usd: Option<u64>) -> Self {
        Self {
            tokens,
            cost_micros_usd,
        }
    }

    fn allows(self, tokens: u64, cost_micros_usd: u64) -> bool {
        self.tokens.is_none_or(|limit| tokens <= limit)
            && self
                .cost_micros_usd
                .is_none_or(|limit| cost_micros_usd <= limit)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BudgetLease {
    id: u64,
    reserved: UsageEstimate,
    request_budget: RequestBudget,
}

impl BudgetLease {
    pub fn new(id: u64, reserved: UsageEstimate, request_budget: RequestBudget) -> Self {
        Self {
            id,
            reserved,
            request_budget,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn reserved(&self) -> UsageEstimate {
        self.reserved
    }

    pub fn request_budget(&self) -> RequestBudget {
        self.request_budget
    }
}

pub trait BudgetManager: Send + Sync + 'static {
    fn remaining(&self, extensions: &RequestExtensions) -> Remaining;
    fn reserve(
        &self,
        extensions: &RequestExtensions,
        estimate: &UsageEstimate,
        request_budget: RequestBudget,
    ) -> Result<BudgetLease, AgentError>;
    fn record_used(&self, lease: BudgetLease, usage: Usage) -> Result<(), AgentError>;
}

impl<T> BudgetManager for Arc<T>
where
    T: BudgetManager + ?Sized,
{
    fn remaining(&self, extensions: &RequestExtensions) -> Remaining {
        (**self).remaining(extensions)
    }

    fn reserve(
        &self,
        extensions: &RequestExtensions,
        estimate: &UsageEstimate,
        request_budget: RequestBudget,
    ) -> Result<BudgetLease, AgentError> {
        (**self).reserve(extensions, estimate, request_budget)
    }

    fn record_used(&self, lease: BudgetLease, usage: Usage) -> Result<(), AgentError> {
        (**self).record_used(lease, usage)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharedPoolBudgetOptions {
    pub capacity_tokens: u64,
    pub capacity_cost_micros_usd: u64,
    pub stop_threshold_tokens: u64,
    pub stop_threshold_cost_micros_usd: u64,
}

impl Default for SharedPoolBudgetOptions {
    fn default() -> Self {
        Self {
            capacity_tokens: u64::MAX,
            capacity_cost_micros_usd: u64::MAX,
            stop_threshold_tokens: 0,
            stop_threshold_cost_micros_usd: 0,
        }
    }
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum SharedPoolBudgetError {
    #[error(
        "request budget exceeded: requested {requested_tokens} tokens / {requested_cost_micros_usd} micros exceeds tokens={budget_tokens:?}, cost={budget_cost_micros_usd:?}"
    )]
    RequestBudgetExceeded {
        requested_tokens: u64,
        requested_cost_micros_usd: u64,
        budget_tokens: Option<u64>,
        budget_cost_micros_usd: Option<u64>,
    },
    #[error(
        "reserving {requested_tokens} tokens / {requested_cost_micros_usd} micros would cross the stop threshold"
    )]
    ThresholdExceeded {
        requested_tokens: u64,
        requested_cost_micros_usd: u64,
        remaining_tokens: u64,
        remaining_cost_micros_usd: u64,
    },
    #[error("unknown budget lease {lease_id}")]
    UnknownLease { lease_id: u64 },
    #[error("shared budget state poisoned")]
    Poisoned,
}

#[derive(Clone)]
pub struct SharedPoolBudgetManager {
    options: SharedPoolBudgetOptions,
    next_lease_id: Arc<AtomicU64>,
    state: Arc<Mutex<SharedPoolBudgetState>>,
}

#[derive(Debug, Default)]
struct SharedPoolBudgetState {
    committed_tokens: u64,
    committed_cost_micros_usd: u64,
    reserved_tokens: u64,
    reserved_cost_micros_usd: u64,
    leases: BTreeMap<u64, (UsageEstimate, RequestBudget)>,
}

impl SharedPoolBudgetManager {
    pub fn new(options: SharedPoolBudgetOptions) -> Self {
        Self {
            options,
            next_lease_id: Arc::new(AtomicU64::new(1)),
            state: Arc::new(Mutex::new(SharedPoolBudgetState::default())),
        }
    }

    fn remaining_with_state(&self, state: &SharedPoolBudgetState) -> Remaining {
        let tokens = self
            .options
            .capacity_tokens
            .saturating_sub(state.committed_tokens.saturating_add(state.reserved_tokens));
        let cost_micros_usd = self.options.capacity_cost_micros_usd.saturating_sub(
            state
                .committed_cost_micros_usd
                .saturating_add(state.reserved_cost_micros_usd),
        );

        Remaining {
            tokens,
            cost_micros_usd,
            below_threshold: tokens <= self.options.stop_threshold_tokens
                || cost_micros_usd <= self.options.stop_threshold_cost_micros_usd,
        }
    }
}

impl BudgetManager for SharedPoolBudgetManager {
    fn remaining(&self, _extensions: &RequestExtensions) -> Remaining {
        let state = self
            .state
            .lock()
            .map_err(|_| AgentError::budget(SharedPoolBudgetError::Poisoned));
        match state {
            Ok(state) => self.remaining_with_state(&state),
            Err(_) => Remaining {
                tokens: 0,
                cost_micros_usd: 0,
                below_threshold: true,
            },
        }
    }

    fn reserve(
        &self,
        _extensions: &RequestExtensions,
        estimate: &UsageEstimate,
        request_budget: RequestBudget,
    ) -> Result<BudgetLease, AgentError> {
        if !request_budget.allows(estimate.total_tokens, estimate.cost_micros_usd) {
            return Err(AgentError::budget(
                SharedPoolBudgetError::RequestBudgetExceeded {
                    requested_tokens: estimate.total_tokens,
                    requested_cost_micros_usd: estimate.cost_micros_usd,
                    budget_tokens: request_budget.tokens,
                    budget_cost_micros_usd: request_budget.cost_micros_usd,
                },
            ));
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| AgentError::budget(SharedPoolBudgetError::Poisoned))?;
        let remaining = self.remaining_with_state(&state);

        let remaining_after_tokens = remaining.tokens.saturating_sub(estimate.total_tokens);
        let remaining_after_cost = remaining
            .cost_micros_usd
            .saturating_sub(estimate.cost_micros_usd);
        let denied = estimate.total_tokens > remaining.tokens
            || estimate.cost_micros_usd > remaining.cost_micros_usd
            || remaining_after_tokens < self.options.stop_threshold_tokens
            || remaining_after_cost < self.options.stop_threshold_cost_micros_usd;

        if denied {
            return Err(AgentError::budget(
                SharedPoolBudgetError::ThresholdExceeded {
                    requested_tokens: estimate.total_tokens,
                    requested_cost_micros_usd: estimate.cost_micros_usd,
                    remaining_tokens: remaining.tokens,
                    remaining_cost_micros_usd: remaining.cost_micros_usd,
                },
            ));
        }

        let id = self.next_lease_id.fetch_add(1, Ordering::Relaxed);
        state.reserved_tokens = state.reserved_tokens.saturating_add(estimate.total_tokens);
        state.reserved_cost_micros_usd = state
            .reserved_cost_micros_usd
            .saturating_add(estimate.cost_micros_usd);
        state.leases.insert(id, (*estimate, request_budget));

        Ok(BudgetLease::new(id, *estimate, request_budget))
    }

    fn record_used(&self, lease: BudgetLease, usage: Usage) -> Result<(), AgentError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| AgentError::budget(SharedPoolBudgetError::Poisoned))?;
        let Some((reserved, request_budget)) = state.leases.remove(&lease.id) else {
            return Err(AgentError::budget(SharedPoolBudgetError::UnknownLease {
                lease_id: lease.id,
            }));
        };

        state.reserved_tokens = state.reserved_tokens.saturating_sub(reserved.total_tokens);
        state.reserved_cost_micros_usd = state
            .reserved_cost_micros_usd
            .saturating_sub(reserved.cost_micros_usd);
        state.committed_tokens = state.committed_tokens.saturating_add(usage.total_tokens);
        state.committed_cost_micros_usd = state
            .committed_cost_micros_usd
            .saturating_add(usage.cost_micros_usd);

        if !request_budget.allows(usage.total_tokens, usage.cost_micros_usd) {
            return Err(AgentError::budget(
                SharedPoolBudgetError::RequestBudgetExceeded {
                    requested_tokens: usage.total_tokens,
                    requested_cost_micros_usd: usage.cost_micros_usd,
                    budget_tokens: request_budget.tokens,
                    budget_cost_micros_usd: request_budget.cost_micros_usd,
                },
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared_pool_error(err: &AgentError) -> &SharedPoolBudgetError {
        match err {
            AgentError::Budget(source) => source
                .downcast_ref::<SharedPoolBudgetError>()
                .expect("shared pool budget error source"),
            other => panic!("expected budget error, got {other}"),
        }
    }

    #[test]
    fn shared_pool_reserves_and_refunds_difference() {
        let manager = SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
            capacity_tokens: 100,
            capacity_cost_micros_usd: 1_000,
            stop_threshold_tokens: 10,
            stop_threshold_cost_micros_usd: 100,
        });

        let extensions = RequestExtensions::new();
        let lease = manager
            .reserve(
                &extensions,
                &UsageEstimate {
                    total_tokens: 20,
                    cost_micros_usd: 200,
                    ..UsageEstimate::zero()
                },
                RequestBudget::unlimited(),
            )
            .unwrap();
        assert_eq!(manager.remaining(&extensions).tokens, 80);

        manager
            .record_used(
                lease,
                Usage {
                    total_tokens: 12,
                    cost_micros_usd: 120,
                    ..Usage::zero()
                },
            )
            .unwrap();

        let remaining = manager.remaining(&extensions);
        assert_eq!(remaining.tokens, 88);
        assert_eq!(remaining.cost_micros_usd, 880);
    }

    #[test]
    fn shared_pool_blocks_when_threshold_would_be_crossed() {
        let manager = SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
            capacity_tokens: 100,
            capacity_cost_micros_usd: 1_000,
            stop_threshold_tokens: 10,
            stop_threshold_cost_micros_usd: 0,
        });

        let err = manager
            .reserve(
                &RequestExtensions::new(),
                &UsageEstimate {
                    total_tokens: 91,
                    ..UsageEstimate::zero()
                },
                RequestBudget::unlimited(),
            )
            .unwrap_err();

        assert!(matches!(
            shared_pool_error(&err),
            SharedPoolBudgetError::ThresholdExceeded { .. }
        ));
    }

    #[test]
    fn request_budget_can_restrict_reservation() {
        let manager = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());

        let err = manager
            .reserve(
                &RequestExtensions::new(),
                &UsageEstimate {
                    total_tokens: 32,
                    ..UsageEstimate::zero()
                },
                RequestBudget::from_tokens(16),
            )
            .unwrap_err();

        assert!(matches!(
            shared_pool_error(&err),
            SharedPoolBudgetError::RequestBudgetExceeded {
                requested_tokens: 32,
                budget_tokens: Some(16),
                ..
            }
        ));
    }

    #[test]
    fn request_budget_can_fail_after_actual_usage_is_higher_than_estimate() {
        let manager = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());

        let lease = manager
            .reserve(
                &RequestExtensions::new(),
                &UsageEstimate {
                    total_tokens: 8,
                    ..UsageEstimate::zero()
                },
                RequestBudget::from_tokens(10),
            )
            .unwrap();

        let err = manager
            .record_used(
                lease,
                Usage {
                    total_tokens: 12,
                    ..Usage::zero()
                },
            )
            .unwrap_err();

        assert!(matches!(
            shared_pool_error(&err),
            SharedPoolBudgetError::RequestBudgetExceeded {
                requested_tokens: 12,
                budget_tokens: Some(10),
                ..
            }
        ));
    }
}

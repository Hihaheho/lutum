use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;

use sqlite_agent::{
    TransactionMode, WriteDecision, WritePreview,
    hooks::{ApproveModeRequest, ApproveWrite, GetTransactionMode},
};

// ---------------------------------------------------------------------------
// TuiApprover — bridges the approve_write hook to the TUI modal via channels
// ---------------------------------------------------------------------------

pub struct TuiApprover {
    pub preview_tx: mpsc::Sender<WritePreview>,
    pub decision_rx: tokio::sync::Mutex<mpsc::Receiver<WriteDecision>>,
}

impl TuiApprover {
    pub fn new(
        preview_tx: mpsc::Sender<WritePreview>,
        decision_rx: mpsc::Receiver<WriteDecision>,
    ) -> Self {
        Self {
            preview_tx,
            decision_rx: tokio::sync::Mutex::new(decision_rx),
        }
    }
}

impl ApproveWrite for TuiApprover {
    async fn call(&self, preview: WritePreview, _last: Option<WriteDecision>) -> WriteDecision {
        // Send preview to TUI for modal display
        if self.preview_tx.send(preview).await.is_err() {
            return WriteDecision::Reject("TUI channel closed".to_string());
        }
        // Wait for the user's decision
        self.decision_rx
            .lock()
            .await
            .recv()
            .await
            .unwrap_or_else(|| WriteDecision::Reject("TUI channel closed".to_string()))
    }
}

// ---------------------------------------------------------------------------
// TuiModeRequestApprover — bridges approve_mode_request hook to TUI modal
// ---------------------------------------------------------------------------

pub struct TuiModeRequestApprover {
    pub request_tx: mpsc::Sender<String>,
    pub decision_rx: tokio::sync::Mutex<mpsc::Receiver<bool>>,
    pub mode: Arc<RwLock<TransactionMode>>,
}

impl TuiModeRequestApprover {
    pub fn new(
        request_tx: mpsc::Sender<String>,
        decision_rx: mpsc::Receiver<bool>,
        mode: Arc<RwLock<TransactionMode>>,
    ) -> Self {
        Self {
            request_tx,
            decision_rx: tokio::sync::Mutex::new(decision_rx),
            mode,
        }
    }
}

impl ApproveModeRequest for TuiModeRequestApprover {
    async fn call(&self, reason: String, _last: Option<bool>) -> bool {
        if self.request_tx.send(reason).await.is_err() {
            return false;
        }
        let granted = self.decision_rx.lock().await.recv().await.unwrap_or(false);
        if granted {
            *self.mode.write().unwrap() = TransactionMode::Writable;
        }
        granted
    }
}

// ---------------------------------------------------------------------------
// TuiModeSource — reads transaction mode from a shared RwLock toggled by Tab
// ---------------------------------------------------------------------------

pub struct TuiModeSource {
    pub mode: Arc<RwLock<TransactionMode>>,
}

impl GetTransactionMode for TuiModeSource {
    async fn call(&self, _last: Option<TransactionMode>) -> TransactionMode {
        *self.mode.read().unwrap()
    }
}

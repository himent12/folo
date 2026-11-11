//! A one-value channel whose receiver is a `Future` (what `prompt_for_paths` returns, awaited by the app
//! as `rx.await` giving `Result<T, Canceled>`, as GPUI's `futures::channel::oneshot`).

use parking_lot::Mutex;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    waker: Option<Waker>,
    sender_gone: bool,
}

/// Creates a sender and receiver for one value.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new(Mutex::new(State { value: None, waker: None, sender_gone: false }));
    (Sender { state: state.clone() }, Receiver { state })
}

/// The sending half: send once, or drop to cancel.
pub struct Sender<T> {
    state: Arc<Mutex<State<T>>>,
}

impl<T> Sender<T> {
    /// Sends the value. Fails (giving it back) if the receiver is gone.
    pub fn send(self, value: T) -> Result<(), T> {
        if Arc::strong_count(&self.state) == 1 {
            return Err(value);
        }
        let waker = {
            let mut state = self.state.lock();
            state.value = Some(value);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut state = self.state.lock();
            state.sender_gone = true;
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The receiving half; awaiting it gives the value, or `Canceled` if the sender was dropped first.
pub struct Receiver<T> {
    state: Arc<Mutex<State<T>>>,
}

impl<T> Receiver<T> {
    /// The value if it has arrived (`Ok(None)`: not yet).
    pub fn try_recv(&mut self) -> Result<Option<T>, Canceled> {
        let mut state = self.state.lock();
        match state.value.take() {
            Some(value) => Ok(Some(value)),
            None if state.sender_gone => Err(Canceled),
            None => Ok(None),
        }
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, Canceled>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock();
        if let Some(value) = state.value.take() {
            return Poll::Ready(Ok(value));
        }
        if state.sender_gone {
            return Poll::Ready(Err(Canceled));
        }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// The sender was dropped without sending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canceled;

impl fmt::Display for Canceled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("oneshot canceled")
    }
}

impl std::error::Error for Canceled {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivers_or_cancels() {
        let (tx, mut rx) = channel();
        assert_eq!(rx.try_recv(), Ok(None));
        tx.send(5).unwrap();
        assert_eq!(rx.try_recv(), Ok(Some(5)));

        let (tx, mut rx) = channel::<u8>();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(Canceled));

        let (tx, rx) = channel();
        drop(rx);
        assert_eq!(tx.send(1), Err(1));
    }
}

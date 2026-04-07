use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Score(f32);

impl Score {
    /// Creates a score without changing the input.
    ///
    /// ```
    /// use lutum_eval::{Score, ScoreRangeError};
    ///
    /// assert_eq!(Score::try_new(0.8).unwrap().value(), 0.8);
    /// assert_eq!(Score::try_new(1.1), Err(ScoreRangeError));
    /// assert_eq!(Score::try_new(f32::NAN), Err(ScoreRangeError));
    /// ```
    pub fn try_new(value: f32) -> Result<Self, ScoreRangeError> {
        if value.is_nan() || !(0.0..=1.0).contains(&value) {
            return Err(ScoreRangeError);
        }

        Ok(Self(value))
    }

    /// Creates a score by clamping the input into `0.0..=1.0`.
    ///
    /// `NaN` is treated as `0.0`.
    ///
    /// ```
    /// use lutum_eval::Score;
    ///
    /// assert_eq!(Score::new_clamped(0.8).value(), 0.8);
    /// assert_eq!(Score::new_clamped(2.0).value(), 1.0);
    /// assert_eq!(Score::new_clamped(-1.0).value(), 0.0);
    /// assert_eq!(Score::new_clamped(f32::NAN).value(), 0.0);
    /// ```
    pub fn new_clamped(value: f32) -> Self {
        if value.is_nan() {
            return Self(0.0);
        }

        Self(value.clamp(0.0, 1.0))
    }

    pub const fn pass() -> Self {
        Self(1.0)
    }

    pub const fn fail() -> Self {
        Self(0.0)
    }

    pub fn value(self) -> f32 {
        self.0
    }

    /// Flips a score so higher becomes lower and vice versa.
    ///
    /// ```
    /// use lutum_eval::Score;
    ///
    /// assert_eq!(Score::try_new(0.2).unwrap().inverse(), Score::try_new(0.8).unwrap());
    /// assert_eq!(Score::pass().inverse(), Score::fail());
    /// ```
    pub fn inverse(self) -> Self {
        Self(1.0 - self.0)
    }
}

impl From<bool> for Score {
    fn from(value: bool) -> Self {
        if value { Self::pass() } else { Self::fail() }
    }
}

#[derive(Debug, Clone, Copy, Eq, Error, PartialEq)]
#[error("score must be within 0.0..=1.0 and not NaN")]
pub struct ScoreRangeError;

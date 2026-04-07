use crate::Score;

pub trait Objective<R> {
    type Error;

    fn score(&self, report: &R) -> Result<Score, Self::Error>;
}

pub trait ObjectiveExt<R>: Objective<R> + Sized {
    fn invert(self) -> InvertObjective<Self> {
        InvertObjective { objective: self }
    }

    fn map_error<F>(self, map: F) -> MapObjectiveError<Self, F> {
        MapObjectiveError {
            objective: self,
            map,
        }
    }
}

impl<R, T> ObjectiveExt<R> for T where T: Objective<R> + Sized {}

pub fn maximize<R, F>(project: F) -> Maximize<F>
where
    F: Fn(&R) -> Score,
{
    Maximize { project }
}

pub fn minimize<R, F>(project: F) -> Minimize<F>
where
    F: Fn(&R) -> Score,
{
    Minimize { project }
}

pub fn pass_fail<R, F>(project: F) -> PassFailObjective<F>
where
    F: Fn(&R) -> bool,
{
    PassFailObjective { project }
}

pub struct Maximize<F> {
    project: F,
}

impl<R, F> Objective<R> for Maximize<F>
where
    F: Fn(&R) -> Score,
{
    type Error = core::convert::Infallible;

    fn score(&self, report: &R) -> Result<Score, Self::Error> {
        Ok((self.project)(report))
    }
}

pub struct Minimize<F> {
    project: F,
}

impl<R, F> Objective<R> for Minimize<F>
where
    F: Fn(&R) -> Score,
{
    type Error = core::convert::Infallible;

    fn score(&self, report: &R) -> Result<Score, Self::Error> {
        Ok((self.project)(report).inverse())
    }
}

pub struct PassFailObjective<F> {
    project: F,
}

impl<R, F> Objective<R> for PassFailObjective<F>
where
    F: Fn(&R) -> bool,
{
    type Error = core::convert::Infallible;

    fn score(&self, report: &R) -> Result<Score, Self::Error> {
        Ok(Score::from((self.project)(report)))
    }
}

pub struct InvertObjective<O> {
    objective: O,
}

impl<R, O> Objective<R> for InvertObjective<O>
where
    O: Objective<R>,
{
    type Error = O::Error;

    fn score(&self, report: &R) -> Result<Score, Self::Error> {
        self.objective.score(report).map(Score::inverse)
    }
}

pub struct MapObjectiveError<O, F> {
    objective: O,
    map: F,
}

impl<R, O, F, E> Objective<R> for MapObjectiveError<O, F>
where
    O: Objective<R>,
    F: Fn(O::Error) -> E,
{
    type Error = E;

    fn score(&self, report: &R) -> Result<Score, Self::Error> {
        self.objective.score(report).map_err(&self.map)
    }
}

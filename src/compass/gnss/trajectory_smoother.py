"""Fixed-interval Rauch-Tung-Striebel smoothing for PPP motion states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RTSSnapshot:
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    predicted_state: np.ndarray | None
    predicted_covariance: np.ndarray | None
    transition: np.ndarray | None
    valid: bool
    transition_valid: bool


def rts_smooth(snapshots: Iterable[RTSSnapshot]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Smooth each continuous valid arc without crossing resets or failed epochs."""
    records = list(snapshots)
    states = [record.filtered_state.copy() for record in records]
    covariances = [record.filtered_covariance.copy() for record in records]
    for k in range(len(records) - 2, -1, -1):
        current, following = records[k], records[k + 1]
        if not current.valid or not following.valid or not following.transition_valid:
            continue
        if following.transition is None or following.predicted_state is None or following.predicted_covariance is None:
            continue
        try:
            gain = np.linalg.solve(
                following.predicted_covariance,
                following.transition @ current.filtered_covariance,
            ).T
        except np.linalg.LinAlgError:
            continue
        states[k] = current.filtered_state + gain @ (states[k + 1] - following.predicted_state)
        covariance = current.filtered_covariance + gain @ (
            covariances[k + 1] - following.predicted_covariance
        ) @ gain.T
        covariances[k] = (covariance + covariance.T) / 2.0
    return states, covariances


def covariance_intersection(
    first_state: np.ndarray, first_covariance: np.ndarray,
    second_state: np.ndarray, second_covariance: np.ndarray,
    weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse correlated forward/reverse estimates without assuming independence."""
    first_information=np.linalg.pinv(first_covariance)
    second_information=np.linalg.pinv(second_covariance)
    information=weight*first_information+(1.0-weight)*second_information
    covariance=np.linalg.pinv(information)
    state=covariance@(
        weight*first_information@first_state+(1.0-weight)*second_information@second_state
    )
    return state,(covariance+covariance.T)/2.0


def trajectory_multipass(
    times: Iterable[float], measurements: Iterable[np.ndarray],
    measurement_covariances: Iterable[np.ndarray], valid: Iterable[bool],
    jerk_noise: float = 1.5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run a standard CA Kalman pass followed by RTS smoothing on a fused trajectory."""
    times=list(times);measurements=[np.asarray(v,float) for v in measurements]
    measurement_covariances=[np.asarray(v,float) for v in measurement_covariances]
    valid=list(valid);records=[];x=None;P=None;last_time=None
    for time,z,R,is_valid in zip(times,measurements,measurement_covariances,valid):
        transition_valid=False;predicted_state=predicted_covariance=transition=None
        if not is_valid:
            records.append(RTSSnapshot(z,R,None,None,None,False,False));x=P=last_time=None
            continue
        if x is None:
            x=z.copy();P=R.copy()
        else:
            dt=time-last_time
            transition=np.eye(9);transition[:3,3:6]=np.eye(3)*dt;transition[:3,6:9]=np.eye(3)*dt*dt/2;transition[3:6,6:9]=np.eye(3)*dt
            h=abs(dt);direction=1.0 if dt>=0.0 else -1.0;q=jerk_noise**2
            block=q*np.array([[h**5/20,direction*h**4/8,h**3/6],[direction*h**4/8,h**3/3,direction*h**2/2],[h**3/6,direction*h**2/2,h]])
            Q=np.zeros((9,9))
            for axis in range(3):Q[np.ix_([axis,axis+3,axis+6],[axis,axis+3,axis+6])]=block
            x=transition@x;P=transition@P@transition.T+Q
            predicted_state=x.copy();predicted_covariance=P.copy();transition_valid=True
            S=P+R;K=np.linalg.solve(S,P).T;x=x+K@(z-x);A=np.eye(9)-K;P=A@P@A.T+K@R@K.T;P=(P+P.T)/2
        records.append(RTSSnapshot(x.copy(),P.copy(),predicted_state,predicted_covariance,transition,is_valid,transition_valid));last_time=time
    return rts_smooth(records)

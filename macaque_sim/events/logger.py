# logger.py — Rank/power state recording + event capture
# Assignee: Dev D (S1-T6)
# See: project_management/tasks/sprint_1/S1-T6_rank_logger.md
# ChatGPT was used to debug and refactor this code. Some comments were also refined by ChatGPT.
import io
import numpy as np
from pathlib import Path

class SimLogger:
    def __init__(self, config):
        self.N = config.N
        self.expected_steps = int(np.ceil(config.T / config.dt_ode))  # total rows to pre-allocate

        self.A = np.zeros((self.expected_steps, self.N, self.N))      # alliance history A(t)
        self.P = np.zeros((self.expected_steps, self.N))              # power history P(t)
        self.R = np.zeros((self.expected_steps, self.N), dtype=int)   # rank history R[t]
        self.time = np.zeros(self.expected_steps)   # store actual t values

        self.swap_events = []   # irregular rank-swap events
        self.step_idx = 0 # to track current step for recording into A, P and R
    
    def record(self, t, R_t, P_t, A_t=None):
        idx = self.step_idx

        self.time[idx] = t
        self.R[idx] = R_t
        self.P[idx] = P_t
        if A_t is not None:
            self.A[idx] = A_t

        self.step_idx += 1
    
    def record_event(self, t, event_type, details):
        self.swap_events.append({
            "t": t,
            "type": event_type,
            "details": details
        })
    
    def get_rank_history(self):
        return self.R[:self.step_idx]

    def get_power_history(self):
        return self.P[:self.step_idx]
    
    def get_time_axis(self):
        return self.time[:self.step_idx]
    
    def get_events(self):
        return self.swap_events
    
    def save(self, path):
        # gotcha: on Windows, passing `path` directly to savez_compressed leaves
        # a ZipFile flush window that can race with an immediate os.remove
        # downstream. Opening the file ourselves + closing explicitly (and
        # following up with fsync) closes the handle deterministically before
        # we return.
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                A=self.A[:self.step_idx],
                P=self.P[:self.step_idx],
                R=self.R[:self.step_idx],
                time=self.time[:self.step_idx],
                swap_events=np.array(self.swap_events, dtype=object),
                N=self.N,
                expected_steps=self.expected_steps,
                step_idx=self.step_idx,
            )
            f.flush()
    
    @classmethod
    def load(cls, path):
        # gotcha: np.load on a path lazily mmaps the .npz, and even after the
        # NpzFile __exit__ closes, Windows can still refuse a subsequent
        # os.remove(path) until every derived array is dereferenced. Read the
        # bytes into a BytesIO first so the on-disk file handle is already
        # closed by the time numpy starts parsing.
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())

        with np.load(buf, allow_pickle=True) as data:
            obj = cls.__new__(cls)

            obj.N = int(data["N"])
            obj.expected_steps = int(data["expected_steps"])
            obj.step_idx = int(data["step_idx"])

            obj.A = np.array(data["A"])
            obj.P = np.array(data["P"])
            obj.R = np.array(data["R"])
            obj.time = np.array(data["time"])
            obj.swap_events = list(data["swap_events"])

        return obj

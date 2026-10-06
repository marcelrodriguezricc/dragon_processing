from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_QC = {
    'roll_cutoff' : 1000000,
    'pitch_cutoff' : 1000000,
    'correlation_cutoff' : 50,
    'HR_correlation_cutoff' : 70,
    'amplitude_cutoff' : 30,
    'err_vel_cutoff' : 0.2,
    'bottom_track_fom_cutoff' : 10000,
    'declination': None,
}

@dataclass
class Config:
    raw: str | Path
    proc: str | Path
    base_path: str | Path | None = None
    roll_cutoff: float = DEFAULT_QC['roll_cutoff']
    pitch_cutoff: float = DEFAULT_QC['pitch_cutoff']
    correlation_cutoff: float = DEFAULT_QC['correlation_cutoff']
    HR_correlation_cutoff: float = DEFAULT_QC['HR_correlation_cutoff']
    amplitude_cutoff: float = DEFAULT_QC['amplitude_cutoff']
    err_vel_cutoff : float = DEFAULT_QC['err_vel_cutoff']
    bottom_track_fom_cutoff : float = DEFAULT_QC['bottom_track_fom_cutoff']
    declination : float = DEFAULT_QC['declination']
    aris_frequency : float = 3
    x0 : float=0
    y0 : float=0
    cast_notes : dict[str, dict[str,dict]] | None = None # {deployment : {cast_num : {'time_start': time, 'time_middle':time, 'time_end': time, 'description', str}}}

    def __post_init__(self) -> None:
        self.raw = Path(self.raw)
        self.proc = Path(self.proc)

        if self.base_path is not None:
            self.base_path = Path(self.base_path)

    def _resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path

        if self.base_path is None:
            raise ValueError(
                f"base_path is required when '{path}' is a relative path"
            )

        return self.base_path / path

    @property
    def raw_path(self) -> Path:
        return self._resolve(self.raw)

    @property
    def proc_path(self) -> Path:
        return self._resolve(self.proc)

    @property
    def qc(self) -> dict[str,float]:
        return {
            k: getattr(self,k) for k in (
                'roll_cutoff',
                'pitch_cutoff',
                'correlation_cutoff',
                'HR_correlation_cutoff',
                'amplitude_cutoff',
                'err_vel_cutoff',
                'bottom_track_fom_cutoff',
                'declination',
            )
            }
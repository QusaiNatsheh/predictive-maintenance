"""Replays held-out engines (never used for training) one cycle at a time, as if their sensors were live."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Machine:
    ref: str
    data: pd.DataFrame  # full run-to-failure history of one engine, sorted by cycle
    position: int = 1  # number of cycles "already happened"

    @property
    def life(self) -> int:
        return int(self.data["cycle"].max())

    @property
    def failed(self) -> bool:
        return self.position >= self.life

    def history(self) -> pd.DataFrame:
        return self.data[self.data["cycle"] <= self.position]


@dataclass
class Fleet:
    machines: dict[str, Machine] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, units: list[int]) -> "Fleet":
        fleet = cls()
        for u in units:
            d = df[df["unit"] == u].sort_values("cycle").reset_index(drop=True)
            ref = f"ENG-{u:03d}"
            fleet.machines[ref] = Machine(ref, d.assign(unit=ref))
        return fleet

    def reset(self, cycles_before_failure: int | None = None, staggered: bool = False) -> None:
        """Start every machine at cycle 1, or a fixed number of cycles before its (hidden) failure.

        staggered=True spreads the machines between 15 and 124 cycles before failure (a repeatable pattern), so a
        demo fleet shows healthy, watched and alerting machines at the same time.
        """
        for i, m in enumerate(self.machines.values()):
            if staggered:
                m.position = max(1, m.life - (15 + (i * 37) % 110))
            else:
                m.position = 1 if cycles_before_failure is None else max(1, m.life - cycles_before_failure)

    def tick(self, cycles: int = 1) -> None:
        for m in self.machines.values():
            m.position = min(m.position + cycles, m.life)

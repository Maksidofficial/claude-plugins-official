"""Paper broker: SimBroker with its state on disk, so a restarted process resumes exactly."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from regimebot.backtest.costs import CostModel
from regimebot.engine import OrderAction
from regimebot.exec.broker import Fill, SimBroker


class PaperBroker(SimBroker):
    def save(self, path: Path) -> None:
        d = {
            "cash": self.cash,
            "costs": asdict(self.costs),
            "qty": self.qty,
            "entry_price": self.entry_price,
            "stop": self.stop,
            "take_profit": self.take_profit,
            "fills": [asdict(f) for f in self.fills],
            "pending": [[asdict(o), s] for o, s in self._pending],
            "entry_state": self._entry_state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, sort_keys=True))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> PaperBroker:
        d = json.loads(path.read_text())
        b = cls(cash=d["cash"], costs=CostModel(**d["costs"]))
        b.qty, b.entry_price = d["qty"], d["entry_price"]
        b.stop, b.take_profit = d["stop"], d["take_profit"]
        b.fills = [Fill(**f) for f in d["fills"]]
        b._pending = [(OrderAction(**o), s) for o, s in d["pending"]]
        b._entry_state = d["entry_state"]
        return b

"""Immutable audit trail system for regulatory compliance and forensic analysis.

Provides cryptographically-secure logging of all trading actions with
tamper-evident storage suitable for regulatory audits.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger


@dataclass
class AuditEntry:
    """Single audit trail entry."""

    timestamp: str
    entry_id: int
    event_type: str  # 'order', 'fill', 'cancel', 'position', 'balance', 'config', 'alert'
    action: str
    symbol: Optional[str]
    details: Dict[str, Any]
    user_id: str = "system"
    previous_hash: str = ""
    entry_hash: str = ""


class AuditTrail:
    """Immutable audit trail with cryptographic verification.

    Each entry contains a hash of the previous entry, creating a blockchain-like
    chain that makes tampering evident.

    Args:
        storage_path: Directory to store audit log files
        max_entries_per_file: Maximum entries before rotating to new file
    """

    def __init__(
        self,
        storage_path: str = "./data/audit",
        max_entries_per_file: int = 10000,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.max_entries_per_file = max_entries_per_file
        self.current_file_index = 0
        self.entry_counter = 0
        self.last_hash = ""

        # Load latest hash from existing trail
        self._initialize_from_existing()

        logger.info(f"AuditTrail initialized at {self.storage_path}")

    def log_event(self, event_type: str, data: dict) -> None:
        """Log a generic trading event to the persistent audit trail.

        This is the primary entry point for all trading-lifecycle events.
        Supported *event_type* values include:

        * ``trade_entry``, ``trade_exit``
        * ``sl_placed``, ``tp_placed``, ``sl_hit``, ``tp_hit``
        * ``position_reduced``, ``emergency_close``
        * ``circuit_breaker``, ``mode_switch``, ``error``

        Args:
            event_type: Structured event category string.
            data: Arbitrary key/value payload.  A ``symbol`` key, if
                present, will be extracted and stored at the top level.
        """
        symbol = data.get("symbol")
        action = data.get("action", event_type)
        details = {k: v for k, v in data.items() if k not in ("symbol", "action")}
        self._write_entry(
            event_type=event_type,
            action=action,
            symbol=symbol,
            details=details,
        )

    def log_order(
        self,
        action: str,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float],
        order_id: Optional[str],
        **kwargs,
    ) -> None:
        """Log an order event."""
        details = {
            "order_type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "order_id": order_id,
            **kwargs,
        }

        self._write_entry(
            event_type="order",
            action=action,
            symbol=symbol,
            details=details,
        )

    def log_fill(
        self,
        symbol: str,
        order_id: str,
        filled_amount: float,
        fill_price: float,
        fee: float,
        **kwargs,
    ) -> None:
        """Log an order fill event."""
        details = {
            "order_id": order_id,
            "filled_amount": filled_amount,
            "fill_price": fill_price,
            "fee": fee,
            **kwargs,
        }

        self._write_entry(
            event_type="fill",
            action="filled",
            symbol=symbol,
            details=details,
        )

    def log_position_change(
        self,
        action: str,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        pnl: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log a position change event."""
        details = {
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "pnl": pnl,
            **kwargs,
        }

        self._write_entry(
            event_type="position",
            action=action,
            symbol=symbol,
            details=details,
        )

    def log_balance_change(
        self,
        action: str,
        currency: str,
        amount: float,
        new_balance: float,
        reason: str,
        **kwargs,
    ) -> None:
        """Log a balance change event."""
        details = {
            "currency": currency,
            "amount": amount,
            "new_balance": new_balance,
            "reason": reason,
            **kwargs,
        }

        self._write_entry(
            event_type="balance",
            action=action,
            symbol=None,
            details=details,
        )

    def log_config_change(
        self,
        action: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        **kwargs,
    ) -> None:
        """Log a configuration change."""
        details = {
            "config_key": config_key,
            "old_value": str(old_value),
            "new_value": str(new_value),
            **kwargs,
        }

        self._write_entry(
            event_type="config",
            action=action,
            symbol=None,
            details=details,
        )

    def log_alert(
        self,
        action: str,
        alert_type: str,
        severity: str,
        message: str,
        **kwargs,
    ) -> None:
        """Log an alert event."""
        details = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            **kwargs,
        }

        self._write_entry(
            event_type="alert",
            action=action,
            symbol=None,
            details=details,
        )

    def _write_entry(
        self,
        event_type: str,
        action: str,
        symbol: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        """Write an entry to the audit trail."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        entry = AuditEntry(
            timestamp=timestamp,
            entry_id=self.entry_counter,
            event_type=event_type,
            action=action,
            symbol=symbol,
            details=details,
            previous_hash=self.last_hash,
        )

        # Calculate hash for this entry
        entry.entry_hash = self._calculate_hash(entry)

        # Write to file
        self._append_to_file(entry)

        # Update state
        self.last_hash = entry.entry_hash
        self.entry_counter += 1

        # Check if need to rotate file
        if self.entry_counter % self.max_entries_per_file == 0:
            self.current_file_index += 1
            logger.info(
                f"Rotating audit log file to index {self.current_file_index}"
            )

    def _calculate_hash(self, entry: AuditEntry) -> str:
        """Calculate SHA-256 hash of entry."""
        # Create canonical representation
        hash_input = f"{entry.timestamp}|{entry.entry_id}|{entry.event_type}|{entry.action}|{entry.symbol}|{json.dumps(entry.details, sort_keys=True)}|{entry.previous_hash}"

        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _append_to_file(self, entry: AuditEntry) -> None:
        """Append entry to current audit log file."""
        file_path = self.storage_path / f"audit_log_{self.current_file_index:06d}.jsonl"

        with open(file_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def _initialize_from_existing(self) -> None:
        """Initialize from existing audit logs."""
        log_files = sorted(self.storage_path.glob("audit_log_*.jsonl"))

        if not log_files:
            logger.info("No existing audit logs found, starting fresh")
            return

        # Read last file
        last_file = log_files[-1]
        self.current_file_index = int(last_file.stem.split("_")[-1])

        # Read last entry to get hash and counter
        with open(last_file, "r") as f:
            lines = f.readlines()

        if lines:
            last_entry = json.loads(lines[-1])
            self.last_hash = last_entry["entry_hash"]
            self.entry_counter = last_entry["entry_id"] + 1

        logger.info(
            f"Resumed audit trail from file {self.current_file_index}, "
            f"entry {self.entry_counter}"
        )

    def verify_integrity(self, start_entry: int = 0, end_entry: Optional[int] = None) -> Tuple[bool, List[str]]:
        """Verify integrity of audit trail.

        Args:
            start_entry: First entry ID to verify
            end_entry: Last entry ID to verify (None = all)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        log_files = sorted(self.storage_path.glob("audit_log_*.jsonl"))
        errors = []

        previous_hash = ""
        entry_id = 0

        for file_path in log_files:
            with open(file_path, "r") as f:
                for line in f:
                    entry_data = json.loads(line)
                    entry = AuditEntry(**entry_data)

                    # Skip if outside range
                    if entry.entry_id < start_entry:
                        previous_hash = entry.entry_hash
                        continue

                    if end_entry and entry.entry_id > end_entry:
                        break

                    # Verify entry ID sequence
                    if entry.entry_id != entry_id:
                        errors.append(
                            f"Entry ID mismatch: expected {entry_id}, got {entry.entry_id}"
                        )

                    # Verify hash chain
                    if entry.previous_hash != previous_hash:
                        errors.append(
                            f"Hash chain broken at entry {entry.entry_id}: "
                            f"expected previous_hash {previous_hash}, got {entry.previous_hash}"
                        )

                    # Verify entry hash
                    calculated_hash = self._calculate_hash(entry)
                    if calculated_hash != entry.entry_hash:
                        errors.append(
                            f"Entry hash mismatch at entry {entry.entry_id}: "
                            f"calculated {calculated_hash}, stored {entry.entry_hash}"
                        )

                    previous_hash = entry.entry_hash
                    entry_id += 1

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Audit trail integrity verified: {entry_id} entries")
        else:
            logger.error(f"Audit trail integrity check FAILED: {len(errors)} errors")

        return is_valid, errors

    def query_entries(
        self,
        event_type: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit trail entries.

        Args:
            event_type: Filter by event type
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        results = []
        log_files = sorted(self.storage_path.glob("audit_log_*.jsonl"))

        for file_path in reversed(log_files):  # Most recent first
            with open(file_path, "r") as f:
                for line in reversed(list(f)):
                    entry_data = json.loads(line)
                    entry = AuditEntry(**entry_data)

                    # Apply filters
                    entry_time = datetime.fromisoformat(entry.timestamp.rstrip("Z"))

                    if start_time and entry_time < start_time:
                        continue

                    if end_time and entry_time > end_time:
                        continue

                    if event_type and entry.event_type != event_type:
                        continue

                    if symbol and entry.symbol != symbol:
                        continue

                    results.append(entry)

                    if len(results) >= limit:
                        return results

        return results

    def export_to_csv(self, output_path: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> int:
        """Export audit trail to CSV for analysis.

        Args:
            output_path: Output CSV file path
            start_time: Start time filter
            end_time: End time filter

        Returns:
            Number of entries exported
        """
        import csv

        entries = self.query_entries(
            start_time=start_time,
            end_time=end_time,
            limit=1000000,  # Large limit
        )

        with open(output_path, "w", newline="") as f:
            if not entries:
                return 0

            # Use first entry to determine fields
            fieldnames = [
                "timestamp",
                "entry_id",
                "event_type",
                "action",
                "symbol",
                "details",
                "entry_hash",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                writer.writerow({
                    "timestamp": entry.timestamp,
                    "entry_id": entry.entry_id,
                    "event_type": entry.event_type,
                    "action": entry.action,
                    "symbol": entry.symbol or "",
                    "details": json.dumps(entry.details),
                    "entry_hash": entry.entry_hash,
                })

        logger.info(f"Exported {len(entries)} entries to {output_path}")
        return len(entries)

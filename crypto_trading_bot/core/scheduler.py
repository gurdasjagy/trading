"""Task scheduling for the trading bot using APScheduler."""

import asyncio
from typing import Callable, Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


class TaskScheduler:
    """Wraps APScheduler's AsyncIOScheduler with a convenience API."""

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._jobs: Dict[str, object] = {}

    def start(self) -> None:
        """Start the underlying scheduler."""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("TaskScheduler started.")

    def stop(self) -> None:
        """Shut down the underlying scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("TaskScheduler stopped.")

    def add_interval_job(
        self,
        func: Callable,
        seconds: int,
        job_id: str,
        **kwargs,
    ) -> None:
        """Schedule *func* to run every *seconds* seconds."""
        job = self._scheduler.add_job(
            func,
            trigger=IntervalTrigger(seconds=seconds),
            id=job_id,
            replace_existing=True,
            **kwargs,
        )
        self._jobs[job_id] = job
        logger.info(f"Interval job added: {job_id!r} (every {seconds}s)")

    def add_cron_job(self, func: Callable, cron_expr: str, job_id: str) -> None:
        """Schedule *func* using a cron expression string (e.g. '0 0 * * *')."""
        fields = cron_expr.split()
        if len(fields) != 5:
            raise ValueError(f"cron_expr must have 5 space-separated fields, got: {cron_expr!r}")
        minute, hour, day, month, day_of_week = fields
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
        )
        self._jobs[job_id] = job
        logger.info(f"Cron job added: {job_id!r} ({cron_expr!r})")

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job by ID."""
        try:
            self._scheduler.remove_job(job_id)
            self._jobs.pop(job_id, None)
            logger.info(f"Job removed: {job_id!r}")
        except Exception as exc:
            logger.warning(f"Could not remove job {job_id!r}: {exc}")

    def pause_job(self, job_id: str) -> None:
        """Pause a scheduled job."""
        try:
            self._scheduler.pause_job(job_id)
            logger.info(f"Job paused: {job_id!r}")
        except Exception as exc:
            logger.warning(f"Could not pause job {job_id!r}: {exc}")

    def resume_job(self, job_id: str) -> None:
        """Resume a paused job."""
        try:
            self._scheduler.resume_job(job_id)
            logger.info(f"Job resumed: {job_id!r}")
        except Exception as exc:
            logger.warning(f"Could not resume job {job_id!r}: {exc}")

    def get_job_status(self, job_id: str) -> dict:
        """Return status information for a single job."""
        job = self._scheduler.get_job(job_id)
        if job is None:
            return {"job_id": job_id, "status": "not_found"}
        return {
            "job_id": job_id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
            "status": "paused" if job.next_run_time is None else "scheduled",
        }

    def list_jobs(self) -> list:
        """Return status dicts for all registered jobs."""
        return [self.get_job_status(job_id) for job_id in self._jobs]

    async def run_job_now(self, job_id: str) -> None:
        """Immediately invoke the job's function outside of its schedule."""
        job = self._scheduler.get_job(job_id)
        if job is None:
            logger.warning(f"run_job_now: job {job_id!r} not found")
            return
        func = job.func
        try:
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            logger.info(f"Job {job_id!r} executed on demand.")
        except Exception as exc:
            logger.error(f"Error running job {job_id!r} on demand: {exc}")

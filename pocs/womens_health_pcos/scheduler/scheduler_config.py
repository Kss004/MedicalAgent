import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

# Get DB URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback for local dev if needed, or raise error in prod
    DATABASE_URL = "postgresql://pcos_user:strongpassword@localhost:5432/pcos_db"

# Fix for postgresql schema if needed (APScheduler uses its own tables)
# Job store ensures that even if the server restarts, the schedule is kept.
jobstores = {
    'default': SQLAlchemyJobStore(url=DATABASE_URL, tablename='apscheduler_jobs', tableschema='pcos')
}
executors = {
    'default': ThreadPoolExecutor(10)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 1
}

scheduler = BackgroundScheduler(
    jobstores=jobstores, 
    executors=executors, 
    job_defaults=job_defaults,
    timezone="UTC"
)

def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        print("[Scheduler] Started persistent background scheduler.")

def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        print("[Scheduler] Shutdown complete.")

import sys
import os
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocs.womens_health_pcos.scheduler.scheduler_config import start_scheduler, shutdown_scheduler, scheduler
from pocs.womens_health_pcos.scheduler.jobs import refresh_stale_pages

def test_scheduler():
    print("Testing Scheduler Initialization...")
    try:
        start_scheduler()
        
        # Add a test job
        if not any(job.id == 'test_job' for job in scheduler.get_jobs()):
            scheduler.add_job(
                lambda: print("Test job running"), 
                'interval', 
                seconds=10, 
                id='test_job'
            )
            print("Test job added.")
        
        jobs = scheduler.get_jobs()
        print(f"Active Jobs: {[j.id for j in jobs]}")
        
        if any(j.id == 'test_job' for j in jobs):
            print("SUCCESS: Scheduler is running and jobs are registered.")
        else:
            print("FAILED: Job registration failed.")
            
    except Exception as e:
        print(f"Scheduler Test Failed: {e}")
    finally:
        shutdown_scheduler()

if __name__ == "__main__":
    test_scheduler()

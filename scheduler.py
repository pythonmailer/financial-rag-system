import os
import time
import threading
from datetime import datetime
import requests
import schedule

from ingest import run_ingestion

# ==========================================
# CONFIG
# ==========================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")
READY_URL = f"{BACKEND_URL}/ready"

TICKERS = os.getenv("SCHEDULER_TICKERS", "AAPL").split(",")
FILING_TYPES = os.getenv("SCHEDULER_FILING_TYPES", "10-K,10-Q").split(",")

RUN_TIME = os.getenv("SCHEDULER_TIME", "00:00")  # 24h format
LOCK = threading.Lock()

# ==========================================
# WAIT FOR BACKEND (Docker-safe)
# ==========================================
def wait_for_backend():
    print("⏳ Scheduler waiting for backend readiness...")
    for _ in range(60):
        try:
            r = requests.get(READY_URL, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ready":
                print("✅ Backend ready for scheduler")
                return
        except:
            pass
        time.sleep(2)

    raise RuntimeError("❌ Scheduler cannot reach backend")

# ==========================================
# INGESTION JOB
# ==========================================
def run_job(trigger_text="⏰ Scheduled ingestion triggered"):
    if LOCK.locked():
        print("⚠️ Previous ingestion still running. Skipping this cycle.")
        return

    with LOCK:
        print(f"{trigger_text} → {datetime.utcnow().isoformat()}")

        for ticker in TICKERS:
            ticker = ticker.strip().upper()
            if not ticker:
                continue

            try:
                print(f"🚀 Running ingestion for {ticker}")
                run_ingestion(ticker=ticker, filing_types=FILING_TYPES)
            except Exception as e:
                print(f"❌ Ingestion failed for {ticker}: {e}")

        print("✅ Scheduled ingestion completed")

# ==========================================
# SCHEDULER SETUP
# ==========================================
def start_scheduler():
    wait_for_backend()

    print(f"📅 Scheduler started — runs daily at {RUN_TIME} UTC")
    print(f"📊 Tickers: {TICKERS}")
    print(f"📄 Filing Types: {FILING_TYPES}")

    schedule.every().day.at(RUN_TIME).do(run_job)

    # Run once on startup
    run_job("🚀 Initial ingestion run on service start")

    while True:
        schedule.run_pending()
        time.sleep(30)

# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    start_scheduler()
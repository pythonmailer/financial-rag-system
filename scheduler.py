import os
import time
import threading
from datetime import datetime, timezone

import requests
import schedule

from ingest import run_ingestion

# ==========================================
# CONFIG
# ==========================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")
READY_URL = f"{BACKEND_URL}/ready"

TICKERS = [t.strip().upper() for t in os.getenv("SCHEDULER_TICKERS", "AAPL").split(",") if t.strip()]
FILING_TYPES = [f.strip().upper() for f in os.getenv("SCHEDULER_FILING_TYPES", "10-K,10-Q").split(",") if f.strip()]

RUN_TIME = os.getenv("SCHEDULER_TIME", "00:00")  # UTC time
LOCK = threading.Lock()

# ==========================================
# WAIT FOR BACKEND (Docker-safe)
# ==========================================
def wait_for_backend():
    print("⏳ Scheduler waiting for backend readiness...", flush=True)

    for _ in range(60):
        try:
            r = requests.get(READY_URL, timeout=3)

            if r.status_code == 200:
                try:
                    if r.json().get("status") == "ready":
                        print("✅ Backend ready for scheduler", flush=True)
                        return
                except Exception:
                    pass
        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError("❌ Scheduler cannot reach backend")

# ==========================================
# INGESTION JOB
# ==========================================
def run_job(trigger_text="⏰ Scheduled ingestion triggered"):
    if LOCK.locked():
        print("⚠️ Previous ingestion still running. Skipping.", flush=True)
        return

    with LOCK:
        print(f"{trigger_text} → {datetime.now(timezone.utc).isoformat()}", flush=True)

        for ticker in TICKERS:
            retries = 3
            for attempt in range(1, retries + 1):
                try:
                    print(f"🚀 Ingestion for {ticker} (attempt {attempt})", flush=True)

                    run_ingestion(
                        ticker=ticker,
                        filing_types=FILING_TYPES,
                    )

                    print(f"✅ Ingestion completed for {ticker}", flush=True)
                    break

                except Exception as e:
                    print(f"❌ Ingestion failed for {ticker}: {e}", flush=True)

                    if attempt == retries:
                        print(f"🚨 Giving up on {ticker} after {retries} attempts", flush=True)
                    else:
                        time.sleep(10)

        print("✅ Scheduled ingestion cycle finished", flush=True)

# ==========================================
# SCHEDULER LOOP (UTC-safe)
# ==========================================
def start_scheduler():
    wait_for_backend()

    print(f"📅 Scheduler started — runs daily at {RUN_TIME} UTC", flush=True)
    print(f"📊 Tickers: {TICKERS}", flush=True)
    print(f"📄 Filing Types: {FILING_TYPES}", flush=True)

    # Schedule job
    schedule.every().day.at(RUN_TIME).do(run_job)

    # Initial run on startup
    run_job("🚀 Initial ingestion run on service start")

    while True:
        schedule.run_pending()
        time.sleep(30)

# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    start_scheduler()
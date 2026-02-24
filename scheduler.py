import time
import schedule
from ingest import run_ingestion

def midnight_job():
    print("⏰ [TRIGGER] Midnight check for new AAPL filings...")
    try:
        # Automatically checks and ingests both annual and quarterly reports
        run_ingestion(ticker="AAPL", filing_types=["10-K", "10-Q"])
    except Exception as e:
        print(f"❌ Scheduler Error: {e}")

# Schedule task at midnight (00:00)
schedule.every().day.at("00:00").do(midnight_job)

if __name__ == "__main__":
    print("🚀 Automatic SEC Ingestion Service Started.")
    # Run once on startup to ensure data is fresh
    midnight_job() 
    
    while True:
        schedule.run_pending()
        time.sleep(60)
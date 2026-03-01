import time
import schedule
from ingest import run_ingestion

def midnight_job(text):
    try:
        print(text)
        run_ingestion(ticker="AAPL", filing_types=["10-K", "10-Q"])
    except Exception as e:
        print(f"❌ Scheduler Error: {e}")

schedule.every().day.at("00:00").do(midnight_job, "⏰ [TRIGGER] Midnight check for new filings...")

if __name__ == "__main__":
    midnight_job("🚀 Automatic SEC Ingestion Service Started.") 
    
    while True:
        schedule.run_pending()
        time.sleep(59)
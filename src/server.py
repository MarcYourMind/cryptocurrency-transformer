from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import uvicorn
import json
import glob
from .bot import TradingBot

app = FastAPI(title="GPT Trading Bot API")
bot = TradingBot()

# Serve static files for the dashboard
if not os.path.exists("static"):
    os.makedirs("static")

@app.on_event("startup")
async def startup_event():
    # Attempt to load state if needed
    pass

@app.get("/status")
def get_status():
    return bot.get_status()

@app.post("/start")
def start_bot():
    bot.start()
    return {"message": "Bot starting..."}

@app.post("/stop")
def stop_bot():
    bot.stop()
    return {"message": "Bot stopping..."}

@app.get("/trades")
def get_trades():
    return {
        "active": bot.active_trades,
        "history": bot.history
    }

@app.get("/api/performance")
def get_performance():
    """Get the latest backtest performance data."""
    try:
        files = glob.glob("results/results_*.json")
        if not files:
            return {"error": "No performance data found."}
        
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
        return {
            "metrics": data.get("metrics"),
            "equity_curve": data.get("equity_curve")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return "<h1>Trading Bot Dashboard</h1><p>Dashboard not found. Please create static/index.html</p>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

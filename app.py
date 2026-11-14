import uvicorn
from src.web_service import app

if __name__ == "__main__":
    uvicorn.run("src.web_service:app", host="127.0.0.1", port=8000, reload=True)
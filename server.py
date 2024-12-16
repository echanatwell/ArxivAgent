import logging

import uvicorn

from fastapi.staticfiles import StaticFiles

from src import app

app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Server is starting - homepage will be available at http://localhost:8000/static/index.html")
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
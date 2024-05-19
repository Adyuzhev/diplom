import uvicorn

if __name__ == '__main__':
    uvicorn.run("app:app", port=8090, host="0.0.0.0")

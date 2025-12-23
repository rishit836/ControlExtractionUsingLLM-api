from fastapi import FastAPI,File, HTTPException, UploadFile
from pathlib import Path
import json
import uuid
import shutil
import threading
# control extraction file 
from pdf_processing import *
import time
import os

app = FastAPI()

# storage directory for pdfs file on the server
UPLOAD_DIR = Path("uploads")
# creating the directory if it doesnt exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Tracks the most recently uploaded PDF so /extract_controls can run on it.
LAST_UPLOADED_PDF: Path | None = None

# endpoint for testing the api server is up and running or not
@app.get("/")
def home():
    return {"msg":"running"}

# this returns if the api is running or not, can be accesed by just putting the url in the browser
@app.get("/")
def status_code():
    return {"msg":"the api is up and running"}


# api endpoint -1
# GET-request on server-ip:server-port/extract_controls/{uuid} uuid of the file uploaded must be passed
# example http://127.0.0.1:8000/extract_controls/c530429137b94b7fb11e0a08cdd00a50
# this ensures that multiple files can be uploaded and data can be fetched based on uuid of the file

@app.get("/extract_controls/{uuid_of_file}")
def controls_request(uuid_of_file:str):
    # checking if the thread process exists for the file based on uuid
    status = uuid_of_file in [thread.name for thread in threading.enumerate()]
    if status:
        # if the thread is still running that is extraction is in process it returns the status.
        return{"status":"the extraction is still in process"}
    else:
        # if the extraction is done it just reads the saved data and returns it to the file
        path = os.path.join("extracted_controls",uuid_of_file+".json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)   
            return data


# api endpoint -2
# POST request on server-ip:server-port/upload requires a key of "file" where the value should be the appropriate PDF file
# if the file isnt a pdf it simply returns a error code 400 else status code 200 is returned
# this creates a thread process in the background for extraction of the controls
# NOTE:the request-sender must store the uuid returned for accesing the file control/status
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    # checking if the pdf file is pdf or not
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    global LAST_UPLOADED_PDF

    file_uuid = uuid.uuid4().hex
    # creating a uuid for the file uploaded so it can be indentified the server if there are mutiple pdf files
    upload_path = UPLOAD_DIR / f"{file_uuid}.pdf"
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    # updating the last_uploaded file so /extract_controls can access.
    LAST_UPLOADED_PDF = upload_path

    # starting a thread process for extracting controls
    extraction_thread = threading.Thread(target=extract_controls,args=(LAST_UPLOADED_PDF,file_uuid,),name=file_uuid)
    try:
    # starting the thread
        extraction_thread.start()
        status_extraction="started"
    except Exception as e:
        print(e)
        status_extraction = f"not started. error:{e}"
    
    # returning a json response of 200 back to the client
    return {
        "status": "uploaded",
        "filename": file.filename,
        "uuid":file_uuid,
        "control_extraction_status":status_extraction
    }

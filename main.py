from fastapi import FastAPI,File, HTTPException, UploadFile,WebSocket,WebSocketDisconnect
from pathlib import Path
import uuid
import shutil
# control extraction file 
from pdf_processing import *
import time
import threading
import os
import json
import asyncio
from pydantic import BaseModel



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


# Upload endpoint

@app.post("/user/framework/upload")
async def upload_user(file: UploadFile = File(...)):
    if not os.path.exists("uploads"):
        UPLOAD_DIR.mkdir(parents=True,exist_ok=True)


    global LAST_UPLOADED_PDF
    extension = file.filename.split('.')[1]
    
    file_uuid = uuid.uuid4().hex
    
    # creating a uuid for the file uploaded so it can be indentified the server if there are mutiple pdf files
    upload_path = UPLOAD_DIR / f"{file.filename}"
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


# Status Get Request
@app.get('/user/framework/status/{uuid_of_file}')
def get_status_user(uuid_of_file:str):
    is_processing = uuid_of_file in [thread.name for thread in threading.enumerate()]

    if is_processing:
        return {"status":"processing"}
    else:
        if os.path.exists(os.path.join(output_folder,f'{uuid_of_file}.json')):
            return {"status":"controls extracted"}
        else:
            return{"status":"invalid uuid"}
        



# websocket for exctract_controls
@app.websocket("/user/framework/extract-controls/{uuid_of_file}")
async def websocket_endpoint_user(websocket: WebSocket, uuid_of_file: str):
    # after how many seconds should we send the request to the client.
    refresh_time = 3
    # 1. Accept the connection
    await websocket.accept()
    curr_state = None
    
    try:
        while True:
            # 2. Check if the extraction thread is active
            is_processing = uuid_of_file in [thread.name for thread in threading.enumerate()]
            
            if is_processing:
                # 3. If processing, send status and wait briefly
                if uuid_of_file in status_msg:
                    if curr_state is not None:
                        if curr_state == status_msg[uuid_of_file]:
                            continue

                    curr_state = status_msg[uuid_of_file]
                    await websocket.send_json({
                        "status": "proccessing", 
                        "message": status_msg[uuid_of_file]
                    })
                else:

                    await websocket.send_json({
                        "status": "started", 
                        "message": "The extraction is still in process"
                    })
                # sleep
                await asyncio.sleep(refresh_time) 
                
            else:
                # 4. If thread is gone, check for the result file
                path = os.path.join(output_folder, f"{uuid_of_file}.json")
                
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # 5. Send final data and close connection
                    await websocket.send_json({
                        "status": "completed", 
                        "data": data
                    })
                    await websocket.close()
                    break
                else:
                    # Handle case where thread finished but no file exists (Error)
                    await websocket.send_json({
                        "status": "error", 
                        "message": "Processing finished but no data found."
                    })
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        print(f"Client disconnected for {uuid_of_file}")












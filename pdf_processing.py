import os
from docx import Document
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import pandas as pd
# import PyPDF2
import pypdf
import os
import json
import re
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from collections import OrderedDict

load_dotenv()

# creating a custom ordered dict for storing status message cache, 
# this way the ram doesnt overload under user increase
class status_cache(OrderedDict):
    def __init__(self,limit=5000, *args,**kwargs):
        self.limit = limit
        super().__init__(*args,**kwargs)

    def  __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)

        if len(self) > self.limit:
            self.popitem(last=False)

global collection,status_msg
status_msg = status_cache(10000)
# --------------------------
# 1. Load and Chunk Document
# --------------------------

# defining path constants
output_folder = 'expert/extracted_controls'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def clean_text(text):
    lines = text.split("\n")

    control_blocks = []
    current_block = []

    # Regex for ID (numbers only OR A.12.3)
    control_id_pattern = re.compile(r'\b(?:[A-Z]+(?:\.[A-Z]+)*-?)?\d+(?:\.\d+)*\b')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect start of a new control
        if control_id_pattern.match(stripped):
            if current_block:
                control_blocks.append("\n".join(current_block))
            current_block = [stripped]
        else:
            if current_block:
                current_block.append(stripped)

    if current_block:
        control_blocks.append("\n".join(current_block))

    final_blocks = []

    for block in control_blocks:
        block_lines = block.strip().split("\n")

        # -------------------------
        # 1. CHECK ID
        # -------------------------
        first_line = block_lines[0]
        if not control_id_pattern.match(first_line):
            continue

        # -------------------------
        # 2. CHECK NAME (Title)
        # It can be on SAME line OR NEXT line
        # -------------------------

        name_found = False

        # (A) Try same line (after removing ID)
        first_line_without_id = control_id_pattern.sub("", first_line).strip()
        if re.search(r'[A-Za-z]', first_line_without_id):
            name_found = True

        # (B) Try next line if no name found yet
        if not name_found and len(block_lines) > 1:
            second_line = block_lines[1].strip()
            # Next line should contain alphabetic words (name/title)
            if re.search(r'[A-Za-z]', second_line) and len(second_line.split()) <= 10:
                # Names are usually short (<=10 words)
                name_found = True

        if not name_found:
            continue    # ❌ No title → discard entire block

        # -------------------------
        # 3. CHECK DESCRIPTION
        # Description must contain ≥5 words (after removing ID & name)
        # -------------------------
        # Description starts from either 2nd or 3rd line
        desc_start_index = 1 if first_line_without_id else 2

        description_lines = block_lines[desc_start_index:]
        description_text = " ".join(description_lines).strip()

        # Remove pure titles from description
        # Description must have meaningful text
        if len(description_text.split()) <2:
            continue  # ❌ Not enough meaningful info

        # All 3 components exist → keep block
        final_blocks.append(block)

    return "\n\n".join(final_blocks)


def normalize_docx_spacing(text):
    # Remove spacing between single characters
    # "t o   P I I" → "to PII"
    text = re.sub(r"(?<=\w)\s+(?=\w)", " ", text)

    # Remove double or triple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text


#The following function returns output in list form with cleaning of text

#The following function returns output in String form
def load_docx(path):
    doc = Document(path)
    lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    # Return as a single string (joined by newline)

    return normalize_docx_spacing("\n".join(lines))


def load_pdf_text(file_path):
    """
    Loads and extracts text from a PDF file.
    Returns the extracted text in the form of a list.
    """
    text = []

    # with open(file_path, "rb") as file:
        # reader = PyPDF2.PdfReader(file)
    reader = pypdf.PdfReader(file_path)

    for page in reader.pages:
        text.append(page.extract_text())
    return text

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split("\n")


import pandas as pd

def load_excel(file_path):
    xls = pd.ExcelFile(file_path)
    text = ""

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        text += df.to_string(index=False) + "\n\n"

    return text.split("\n")



def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    

    if ext == ".docx":
        return load_docx(file_path)
    elif ext == ".pdf":
        return load_pdf_text(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext in [".xls", ".xlsx"]:
        return load_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


#Extracting controls from the document of Client


#The following function removes bullets, dots and numbering
def clean_leading_bullets(text):
    if not text:
        return text

    # remove typical bullet characters
    text = re.sub(r"^[\s•·▪○●►▶↳\-\*]+", "", text)

    # remove stray dots like ". " or " . "
    text = re.sub(r"^\s*\.\s*", "", text)
    """
    Removes leading numbering patterns like:
    '8. Text', '8) Text', '(8) Text', '8 - Text', '8: Text'
    """
    return re.sub(r"^\s*\(?\d+\)?\s*[\.\:\-\)]\s*", "", text).strip()

    return text.strip()




semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text1, text2):
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round(float(score), 4)

#--------------------------------
# Creating chunks of the text
#--------------------------------
def chunk_text(text_list, chunk_size=1000):
    chunks = []
    current = ""

    for line in text_list:
        if len(current) + len(line) <= chunk_size:
            current += " " + line
        else:
            chunks.append(current.strip())
            current = line

    if current:
        chunks.append(current.strip())

    return chunks


# --------------------------
# 3. Insert chunks into vector DB
# --------------------------
def insert_into_chroma(collection, chunks):
    ids = []
    for i, chunk in enumerate(chunks):
        ids.append(str(i))
    collection.add(
        documents=chunks,
        ids=ids
    )

# --------------------------
# NEW: Retrieve similar chunks (RAG Retrieval)
# --------------------------
def retrieve_similar_chunks(collection, query, top_k=70):
    """
    Return clean list of chunk strings.
    """
    result = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    try:
        docs = result["documents"][0]
        return [d for d in docs if d.strip()]
    except:
        return []


# --------------------------
# 4. RAG QUERY: Ask LLM to Extract Controls
# --------------------------
client = OpenAI(api_key=os.environ['openai_key'])

import json
# --------------------------
# 4. EXTRACT CONTROL: Ask LLM to Extract Controls  (modified to use a list of strings)
# --------------------------
REGEX = r"\b(?:[A-Z]+(?:\.[A-Z]+)*[-.]?)?\d+(?:\.\d+)*\b"
def extract_controls_full(chunks,uuid_of_file):
    global status_msg

    '''
    # 1. Retrieve top relevant chunks
    rag_chunks = retrieve_similar_chunks(query, top_k=70)

    print("Retrieved chunks = ", len(rag_chunks))
    for i, c in enumerate(rag_chunks):
        print(f"\n--- Chunk {i+1} ---\n{c[:400]}\n")


    # If RAG fails, fallback to all chunks
    if not rag_chunks:
        rag_chunks = chunks
    #The following variable is used in the prompt (for extraction of the controls)
    #This way relevant chunks are used for this purpose

    #rag_chunks = chunks
    combined_context = "\n\n---\n\n".join(rag_chunks)
    '''
    prompt = f"""
You are a strict JSON generator.
Extract ALL compliance controls from the following text.
Do NOT skip ANY control.
Do NOT give anything extra.
Strictly search in the document.

STRICTLY extract all control IDs using the regex ONLY.

Do NOT invent or skip any ID.

STRICTLY extract all control IDs using this regex:
{REGEX!r}

Treat ALL numeric headings (0.1,0.2,0.3,0.4,1.1,1.2,2.3,1.1.2,A.1.2,A.2 etc.) as controls.
Do NOT treat them as TOC headings.

EXTRACT ONLY if all three are present in this sequence- ID, NAME and DESCRIPTION otherwise return null.

DESCRIPTION CAN BE OF 5 TO 1500 characters.

Consider all the controls given till the end.

Use JSON list ONLY:

[
  {{
    "Control_id": "",
    "Control_name": "",
    "Control_type":"",
    "Control_description": ""
  }}
]

TEXT:
{chunks}

Return ONLY JSON. No markdown. No text outside JSON.
"""

    print("extracting controls")
    status_msg[uuid_of_file] = "extracting controls from document."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    prompt = f"""
You are an anlyser/interpreter.
From each of the given compliance controls in the given JSON, analyse and retrieve the deployment points.
Do NOT skip ANY control.


INTERPRET THE TEXT IN THE DESCRIPTION AND GENERATE 5 to 6 POINTS FOR THE DEPLOYMENT (for each control seperatrly).
STORE THE DEPLOYMENT related points under Deployment_points heading.
Deployment points must describe:
- how this control is implemented,
- what actions/steps are required,
- how to operationalize the control.
- any other important point of deployment.
Every point should be numbered.
All the points should be stored in the form of a single string.

Use the following JSON for the above operation:
{response}

After the above retrieval operation, add one more heading in the JSON - Deployment_points and store the deployment points in it.
Use JSON list ONLY:

[
  {{
    "Control_id": "",
    "Control_name": "",
    "Control_type":"",
    "Control_description": "",
    "Deployment_points": ""
  }}
]

TEXT:
{chunks}

Return ONLY JSON. No markdown. No text outside JSON.
"""

    print("extracting deployment points.")
    status_msg[uuid_of_file] = "extracting deployment points"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print("⚠ JSON error")
        print(response.choices[0].message.content)
        return []


def remove_duplicate_controls(controls):
    """
    Removes duplicate controls based on Control_id.
    Keeps only the FIRST occurrence and discards others.
    """
    unique = {}
    cleaned_list = []

    for ctrl in controls:
        cid = ctrl.get("Control_id", "").strip()

        # Skip invalid entries
        if cid == "":
            continue

        # Keep only the first time a Control_id appears
        if cid not in unique:
            unique[cid] = True
            cleaned_list.append(ctrl)

    return cleaned_list
# --------------------------
# 5. RUN PIPELINE
# --------------------------

# --------------------------
#Setup ChromaDB (Vector DB)
# --------------------------


def extract_controls(file_path,uuid_of_file):
    chroma_client = chromadb.PersistentClient(path="chroma_db/")

    # Always reset the collection so old chunks do not pollute results
    try:
        chroma_client.delete_collection("controls_collection")
    except:
        pass

    collection = chroma_client.create_collection(
        name="controls_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Step 1: Load document
    text_content_list = load_document(file_path)


    # Step 2: Chunk text
    chunks = chunk_text(text_content_list)

    # Step 3: Insert into vector DB
    insert_into_chroma(collection,chunks)
    #import time
    #time.sleep(15)
    # Step 4: Run extraction
    #controls = extract_controls_full(chunks)
    #query = "List all compliance control statements, IDs, Names, Types and Descriptions."
    out_path = os.path.join(output_folder,uuid_of_file+".json")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = out_path

    # --- Check if JSON already exists ---
    if os.path.exists(output_file):
        print("JSON file already exists. Loading existing controls...")

        with open(output_file, "r") as f:
            json_text = f.read()

        controls = json.loads(json_text)

    else:
        print("Extracting controls...")

        controls = extract_controls_full(chunks,uuid_of_file)   

        # Convert to JSON text
        json_text = json.dumps(controls, indent=2)

        # Save JSON to file
        with open(output_file, "w") as f:
            f.write(json_text)

        print("JSON saved to:", output_file)

    print("done")


def extract_controls_user(file_path,uuid_of_file):
    chroma_client = chromadb.PersistentClient(path="chroma_db/")

    # Always reset the collection so old chunks do not pollute results
    try:
        chroma_client.delete_collection("controls_collection")
    except:
        pass

    collection = chroma_client.create_collection(
        name="controls_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Step 1: Load document
    text_content_list = load_document(file_path)


    # Step 2: Chunk text
    chunks = chunk_text(text_content_list)

    # Step 3: Insert into vector DB
    insert_into_chroma(collection,chunks)
    #import time
    #time.sleep(15)
    # Step 4: Run extraction
    #controls = extract_controls_full(chunks)
    #query = "List all compliance control statements, IDs, Names, Types and Descriptions."
    out_path = os.path.join(output_folder,uuid_of_file+".json")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = out_path

    # --- Check if JSON already exists ---
    if os.path.exists(output_file):
        print("JSON file already exists. Loading existing controls...")

        with open(output_file, "r") as f:
            json_text = f.read()

        controls = json.loads(json_text)

    else:
        print("Extracting controls...")

        controls = extract_controls_only(chunks,uuid_of_file)   

        # Convert to JSON text
        json_text = json.dumps(controls, indent=2)

        # Save JSON to file
        with open(output_file, "w") as f:
            f.write(json_text)

        print("JSON saved to:", output_file)

    print("done")


def extract_controls_only(chunks,uuid_of_file):
    global status_msg

    '''
    # 1. Retrieve top relevant chunks
    rag_chunks = retrieve_similar_chunks(query, top_k=70)

    print("Retrieved chunks = ", len(rag_chunks))
    for i, c in enumerate(rag_chunks):
        print(f"\n--- Chunk {i+1} ---\n{c[:400]}\n")


    # If RAG fails, fallback to all chunks
    if not rag_chunks:
        rag_chunks = chunks
    #The following variable is used in the prompt (for extraction of the controls)
    #This way relevant chunks are used for this purpose

    #rag_chunks = chunks
    combined_context = "\n\n---\n\n".join(rag_chunks)
    '''
    prompt = f"""
You are a strict JSON generator.
Extract ALL compliance controls from the following text.
Do NOT skip ANY control.
Do NOT give anything extra.
Strictly search in the document.

STRICTLY extract all control IDs using the regex ONLY.

Do NOT invent or skip any ID.

STRICTLY extract all control IDs using this regex:
{REGEX!r}

Treat ALL numeric headings (0.1,0.2,0.3,0.4,1.1,1.2,2.3,1.1.2,A.1.2,A.2 etc.) as controls.
Do NOT treat them as TOC headings.

EXTRACT ONLY if all three are present in this sequence- ID, NAME and DESCRIPTION otherwise return null.

DESCRIPTION CAN BE OF 5 TO 1500 characters.

Consider all the controls given till the end.

Use JSON list ONLY:

[
  {{
    "Control_id": "",
    "Control_name": "",
    "Control_type":"",
    "Control_description": ""
  }}
]

TEXT:
{chunks}

Return ONLY JSON. No markdown. No text outside JSON.
"""

    print("extracting controls")
    status_msg[uuid_of_file] = f"extracting controls from document"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    status_msg[uuid_of_file] = "extraction complete"
 
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print("⚠ JSON error")
        return []


# saving comparsion status
global comparision_status
comparision_status = status_cache(10000)

def get_comparision_dict(user_filename, expert_filename,user_uuid):
    try:
        client_df = pd.DataFrame(pd.read_json(user_filename))
        expert_df = pd.DataFrame(pd.read_json(expert_filename))
        results = []
        comparision_status[str(user_uuid)] = "proccessing"
        
        for client_idx ,client_row in client_df.iterrows():
            c_name = client_row["Control_name"]
            c_desc = client_row["Control_description"]
            best_score = -1
            best_match = None

            for row_idx, row in expert_df[['Control_id','Control_name', 'Control_description','Deployment_points']].iterrows():
                

                f_name = row.get("Control_name", "")
                f_desc = row.get("Control_description", "")

                deployment_pts =row.get("Deployment_points", "")
                framework_text = f"{f_name} {f_desc} {deployment_pts}"
                client_text    = f"{c_name} {c_desc}"

                score = semantic_similarity(client_text, framework_text)
                if score > best_score:
                    best_score = score
                    best_match = row

            
            results.append({
                "User_Document_Control_Name": c_name,
                "User_Document_Control_Description": c_desc,
                "Expert_Framework_Control_Id": best_match.get("Control_id", ""),
                "Expert_Framework_Control_Name": best_match.get("Control_name", ""),
                "Expert_Framework_Control_Description": best_match.get("Control_description", ""),
                "Deployment_Points": best_match.get("Deployment_points", []),
                "Comparison_Score": best_score
            })
        
        # comparision_status[str(user_uuid)] = "done"
        results_df = pd.DataFrame(results)
        if not os.path.exists('comparisions/'):
            os.mkdir('comparisions')
        
        with open(os.path.join('comparisions',user_uuid+'.json'), 'w') as f:
            json.dump(results_df.to_dict(orient='records'), f, indent=4)

        return results_df
    except:
        print("invalid file format returned.")
        comparision_status[str(user_uuid)] = "error"
        
    
def split_deployment_points(deployment_text):
    """
    Splits numbered deployment points into a list.
    Handles formats like:
    1. text
    2) text
    - text
    """
    if not deployment_text:
        return []

    points = re.split(r'\n|\r|(?:(?:^|\s)(?:\d+[\.\)]|-)\s+)', deployment_text)
    points = [p.strip() for p in points if len(p.strip().split()) > 3]
    return points

def compare_deployment_points(comparison_df):
    deployment_results = []

    for _, row in comparison_df.iterrows():

        client_desc = row["User_Document_Control_Description"]
        framework_id = row["Expert_Framework_Control_Id"]
        framework_name = row["Expert_Framework_Control_Name"]

        deployment_points = split_deployment_points(row["Deployment_Points"])

        for idx, point in enumerate(deployment_points, start=1):
            score = semantic_similarity(client_desc, point)

            if score >= 0.75:
                status = "Implemented"
            elif score >= 0.50:
                status = "Partially Implemented"
            else:
                status = "Missing"

            deployment_results.append({
                "Expert_Framework_Control_Id": framework_id,
                "Expert_Framework_Control_Name": framework_name,
                "Deployment_Point_No": idx,
                "Deployment_Point": point,
                "User_Document_Control_Description": client_desc,
                "Similarity_Score": score,
                "Implementation_Status": status
            })

    return pd.DataFrame(deployment_results)

deployment_status = status_cache(limit=10000)
def deployment_pipeline(uuid_user,uuid_expert):
    deployment_status[str(uuid_expert)] = "processing"
    user_file_path = os.path.join(output_folder, uuid_user + '.json')
    expert_file_path = os.path.join(output_folder, uuid_expert + '.json')
    comparision_df = get_comparision_dict(user_file_path,expert_file_path,uuid_user)
    deployment_pointdf = compare_deployment_points(comparision_df)
    if not os.path.exists('deployment_point_compared'):
        os.mkdir('deployment_point_compared')
    with open(os.path.join('deployment_point_compared', uuid_user + '.json'),"w") as f:
        f.write(json.dumps(deployment_pointdf.to_dict(orient='records'),indent=2))
        deployment_status[str(uuid_expert)] = "completed"









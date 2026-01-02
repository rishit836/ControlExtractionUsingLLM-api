import pypdf
import bitsandbytes
import torch
import json
import time
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,pipeline
import os
from huggingface_hub import login
import re
from rag_validation import *
from dotenv import load_dotenv
import threading

load_dotenv()

login(token=os.environ['hf_token'])

global tokenizer,model
tokenizer,model = None,None
def normalize_text(raw_text):

    # checking for empty string incase pdf has some
    if not raw_text:
        return ""

    # splitting lines
    lines = raw_text.split('\n')
    cleaned_lines = []

    # Generic noise patterns found the pdf[nist and iso]
    # defining the list because incase new patterns need to be added based on the pdf file.
    # These are safe to remove from ANY document.
    # We include a length check (len < 30) to ensure we don't accidentally
    # delete a real control that happens to contain the word "Page".
    noise_patterns = [
        r"^Page\s+\d+$",               # Matches "Page 1"
        r"^Page\s+\d+\s+of\s+\d+$",    # Matches "Page 1 of 10"
        r"^\d+\s+of\s+\d+$",           # Matches "1 of 10"
        r"^https?://",                 # URL artifacts often in footers
        r"^www\.",                     # Web links
        r"^\(c\)\s+\d{4}",             # Copyright markers like "(c) 2023"
        r"^Copyright",                 # Copyright word
    ]

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if line is noise
        is_noise = False
        # Only check short lines to be safe. If a line is 100 chars long,
        # it's likely content, even if it has "Page" in it.
        if len(line) < 40:
            for pattern in noise_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_noise = True
                    break

        if not is_noise:
            # Whitespace Normalizaiton
            # Replace non-breaking spaces (\xa0) with normal spaces

            line = line.replace('\xa0', ' ').replace('\t', ' ')

            # Collapse multiple spaces into one (e.g. "Policy    Name" -> "Policy Name")
            line = re.sub(r'\s+', ' ', line)

            cleaned_lines.append(line)

    # Merging all the lines into one text
    # We join with '\n' to preserve the structure.
    # The LLM needs to see the newlines to understand the layout.
    return '\n'.join(cleaned_lines)





# extracting the text from the pdf
def extract_pdf_text(pdf_path):
    # create a reader object
    reader = pypdf.PdfReader(pdf_path)
    text_pages = []

    for page in reader.pages:
        page_cleaned = normalize_text(page.extract_text())
        text_pages.append(page_cleaned)
    
    # return the list full of content/text of pages in the pdf
    return text_pages

# load the model, default model is qwen2.5 but this can be changed in case for future
def load_model(model_id="Qwen/Qwen2.5-7B-Instruct"):
    global tokenizer,model

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Set padding side to 'left' for generation (Important for decoder-only models)
    tokenizer.padding_side = "left"

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer,model

# this is the function which extracts control from each page in the text_page
def extract_controls_from_page(page_text,model,tokenizer):
    # prompt for the llm
    system_prompt = """

You are an expert Compliance Auditor specialized in ISO, NIST, and Regulatory frameworks.

Task: Extract individual definitive controls or regulatory requirements from the provided text.

CRITICAL - ANTI-HALLUCINATION RULES:
1. IGNORE "Related Controls":
   - You will see lines like: "Related Controls: AC-3, AC-5, AC-6..."
   - THESE ARE REFERENCES, NOT DEFINITIONS.
   - NEVER extract an ID from a "Related Controls" line.
   - If an ID appears at the *end* of a text block, it is a reference. Ignore it.

2. IGNORE "Discussion" Sections:
   - Text labeled "Discussion:" is explanatory context. It is NOT the control requirement.
   - Do not extract text from the "Discussion" section as a "control_desc".

3. IGNORE "Enhancements" without full IDs:
   - Items listed as "(1)", "(2)", "(a)" are sub-enhancements.
   - Unless they have the full prefix (e.g., "AC-2(1)"), they are INVALID.
   - In this specific context, if you see "(1) ACCOUNT MANAGEMENT", skip it.

Control ID Identification Logic:
A Valid Control ID must be the **first** significant element on a line.
- NIST Style: Must start with letters, followed by a hyphen and number (e.g., "AC-1", "PM-10").
- Annex A Style: Must look like "A.5.1".
- Section Style: Must look like "Sec. 404".

STRICT INVALID IDs (STOP if you see these):
- "(1)", "(2)", "(a)", "(b)"
- "AC-3, AC-5" (Comma-separated lists are references)
- IDs found inside a sentence.

Extraction Rules:
- If valid controls are found, return ONLY a raw JSON list.
- Fields:
    - "control_id": The exact identifier.
    - "control_title": The heading.
    - "control_desc": The operative "shall/must" statement. if it does,t ten its not a control and dont include that in the list.
- If NO controls are found (e.g., only Discussion text or Enhancements), output strictly: []

Output Format:
- Return ONLY the raw JSON list. No markdown. No explanations.
    """
    # message format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this text:\n\n{page_text}"}
    ]

    # Prepare Inputs
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 4. Generate
    outputs = model.generate(
        input_ids,
        max_new_tokens=4096,      # Limit output size to save time
        do_sample=False,         # temperature=0 (Deterministic)
        temperature=None,        # Must be None if do_sample=False
        top_p=None               # Must be None if do_sample=False
    )

    # 5. Decode
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # returning the response
    return response.strip()

def clean_and_validate_json(json):
    indices_to_remove = []
    seen_ids = set()
    for idx,control in enumerate(json):
        empty_string_type = ['','none',' ',None]
        if control['control_id'].lower() not in seen_ids:
            seen_ids.add(control['control_id'].lower())
        else:
            indices_to_remove.append(idx)
            continue
        if len(control['control_desc'])<3:
            indices_to_remove.append(idx)
            continue
        if control['control_id'].lower() in empty_string_type or control['control_title'].lower() in empty_string_type or control['control_desc'].lower() in empty_string_type:
            indices_to_remove.append(idx)
            continue
        

    json = [element for index, element in enumerate(json) if index not in indices_to_remove]

    

    return json


def extract_controls(pdf_path,uuid_of_file):
    global tokenizer,model
    

    # list of all controls
    all_extracted_data = []

    # for debug purpose incase the control extraction fails at any point we can see what did the model return for cleaning the output more
    responses = []

    # load the data and model
    text_page = extract_pdf_text(pdf_path)
    if tokenizer is None or model is None:
        tokenizer,model = load_model()

    # calculating time per page to see the efficiency can be removed
    total_start_time = time.time()  # Start global timer
    print("Starting extraction...")

    for i, page_text in enumerate(text_page[40:50]):
        print(f"--- Processing Page {i+1} ---")

        # save the time at which the processing starts
        page_start_time = time.time()

        # get the response from the llm on the page text
        raw_response = extract_controls_from_page(page_text,model,tokenizer)

        # the model returns the json list in a code snippet thus we take whatever text is inside that snippet and leave the rest
        clean_json_string = raw_response.replace("```json", "").replace("```", "").strip()

        # save the time at which the model has processed the page
        page_end_time = time.time()
        # calculate the time taken by model to process the page
        elapsed_time = page_end_time - page_start_time
        print(f"   > Time taken: {elapsed_time:.2f} seconds")
        responses.append(clean_json_string)

        # using a try-except clause so incase the model returns a invalid string [ it hasnt found the controls in the file]then we skip the page

        try:
            if not clean_json_string or clean_json_string == "[]":
                print(f"   > No controls found.")
                continue

            data = json.loads(clean_json_string)

            # Verify it's a list
            if isinstance(data, list):
                count = len(data)
                print(f"   > Success! Found {count} controls.")
                        
                all_extracted_data.extend(data)
            else:
                print(f"   > Warning: Model returned valid JSON but not a list.")

        except json.JSONDecodeError:
            print(f"   > Error: Model output invalid JSON.\n   > Raw Output: {raw_response[:50]}...")

    # calculating total time taken by model to process the pdf
    # again for the testing purpose and can be removed

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)


    # cleaning the json file (removing the data which is not a valid control)
    # print(f"Raw controls found: {len(all_extracted_data)}")
    
    # starting the validation process thread in the background
    threading.Thread(target=validate_controls,args=(text_page,all_extracted_data,uuid_of_file,)).start()
    

    if not os.path.exists("extracted_controls/"):
        os.mkdir("extracted_controls")


    with open(f"extracted_controls/{uuid_of_file}.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_data, f, indent=2, ensure_ascii=False)

    print(f"extraction done in {minutes}m {seconds}s.found {len(all_extracted_data)}. Validation Process Started in the Background.")
import pypdf
import bitsandbytes
import torch
import json
import time
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,pipeline
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(os.environ['hugging_face_token'])

import re
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
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
# using shorter prompt for now to remove hallucination
    system_prompt ="""
    Role: Compliance Auditor.

Task: Extract controls into a raw JSON list: [{"control_id", "control_title", "control_desc"}].

Strict Extraction Rules:

    -Valid IDs: Must appear at the start of a line (e.g., AC-1, A.5.1, Sec. 404).
    -Valid Description: longer than 3 words, and must use imperative words.

    note: if any of the above is absent dont include that in the list.

Ignore:
    -"Related Controls" or comma-separated reference lists.
    -"Discussion" sections.
    -Sub-items like "(1)" or "(a)" unless they include the full prefix (e.g., "AC-2(1)").
    -IDs embedded in the middle of sentences.

Validation: control_desc must be >3 words. Exclude entries if any field is missing.

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


d# validating the controls in the json file extracted using llm.
def clean_and_validate_controls(controls_list):
    cleaned_list = []
    
    # --- 1. Regex Patterns for Different Standards ---
    
    # NIST Style: 2+ uppercase letters, hyphen, numbers (e.g., AC-1, AC-2(a))
    nist_pattern = r"^[A-Z]{2,}-\d+(?:\([a-z0-9]+\))?([a-z])?$"
    
    # ISO / PCI / CIS Style: Starts with digit or 'A.', followed by dots (e.g., A.5.1.1, 9.2, 1.1.1)
    dot_notation_pattern = r"^[A-Z]?\d+(?:\.\d+)+([a-z])?$"
    
    # Combined pattern (Matches either)
    valid_id_regex = re.compile(f"({nist_pattern})|({dot_notation_pattern})")

    # --- 2. Blocklist (Stop "Page 10", "Section 4", dates, etc.) ---
    invalid_prefixes = ["page", "section", "chapter", "table", "figure", "appendix", "version", "copyright"]

    # --- 3. Hallucination Filters ---
    hallucination_triggers = [
        "not explicitly stated", 
        "related to", 
        "see control", 
        "this section intentionally left blank",
        "reserved"
    ]

    seen_ids = set()

    for entry in controls_list:
        # skipping invalid json token (highly unlikely)
        if not isinstance(entry, dict):
            continue
            
        # Extract & Clean Strings
        c_id = str(entry.get("control_id", "")).strip()
        c_title = str(entry.get("control_title", "")).strip()
        c_desc = str(entry.get("control_desc", "")).strip()
        
        # Normalize Smart Quotes & Spaces
        replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\xa0": " "}
        for k, v in replacements.items():
            c_id = c_id.replace(k, v)
            c_title = c_title.replace(k, v)
            c_desc = c_desc.replace(k, v)

        # CHECK 1: Missing Data
        if not c_id or not c_title or not c_desc:
            continue

        # CHECK 2: Blocklist (Fast fail for obvious noise)
        if any(c_id.lower().startswith(prefix) for prefix in invalid_prefixes):
            continue

        # CHECK 3: ID Format Validation (Flexible)
        # If it doesn't match NIST or Dot-notation, we skip it.
        # This filters out random headers like "Access Control Policy" getting into the ID field.
        if not valid_id_regex.match(c_id):
            # Fallback: If your extraction is messy, you might loosen this, 
            # but keeping it strict ensures high quality for ISO/NIST.
            continue

        # CHECK 4: Description Quality
        if len(c_desc.split()) < 3: # allow slightly shorter ISO descriptions
            continue
            
        if any(phrase in c_desc.lower() for phrase in hallucination_triggers):
            continue

        # CHECK 5: Deduplication
        # Only keep the first occurrence of an ID to prevent page overlaps duplicating data
        if c_id in seen_ids:
            continue
        seen_ids.add(c_id)

        cleaned_list.append({
            "control_id": c_id,
            "control_title": c_title,
            "control_desc": c_desc
        })

    return cleaned_list


def extract_controls(pdf_path,uuid_of_file):

    # list of all controls
    all_extracted_data = []

    # for debug purpose incase the control extraction fails at any point we can see what did the model return for cleaning the output more
    responses = []

    # load the data and model
    text_page = extract_pdf_text(pdf_path)
    tokenizer,model = load_model()

    # calculating time per page to see the efficiency can be removed
    total_start_time = time.time()  # Start global timer
    print("Starting extraction...")

    for i, page_text in enumerate(text_page):
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
    print(f"Raw controls found: {len(all_extracted_data)}")
    final_clean_data = clean_and_validate_controls(all_extracted_data)
    print(f"Cleaned controls: {len(final_clean_data)}")

    if not os.path.exists("extracted_controls/"):
        os.mkdir("extracted_controls")


    with open(f"extracted_controls/{uuid_of_file}.json", "w", encoding="utf-8") as f:
        json.dump(final_clean_data, f, indent=2, ensure_ascii=False)

    print(f"extraction done in {minutes}m {seconds}s.found {len(final_clean_data)} ({total_duration//len(final_clean_data)}s/per control)")
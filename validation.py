from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import re


def load_model(model_id="Qwen/Qwen2.5-7B-Instruct"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

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

def validate_controls(context,control_id,control_title,control_desc,tokenizer,model):
    validation_template = f"""
    You are a Quality Assurance Auditor. Verify if the following control exists in the source text.
    
    SOURCE TEXT (Reference):
    {context}
    
    CLAIMED CONTROL (To Verify):
    ID: {control_id}
    Title: {control_title}
    Description: {control_desc}
    
    TASK:
    Determine if the "Claimed Control" is explicitly supported by the "Source Text".
    Also make sure the control given isnt just a change made in any other control
    
    OUTPUT FORMAT:
    Return only a JSON object with this format:
    {{
        "is_valid": true/false,
        "reason": "Short explanation citing the text",
        "confidence_score": 1-10
    }}
    """
    messages = [
        {"role": "user", "content": validation_template}
    ]
    # message format
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
    # Determine the length of input tokens so we only decode the new response tokens
    response_tokens = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    # returning the response
    return clean_and_parse_json(response_text)


def clean_and_parse_json(text):
    """Helper to strip markdown code blocks and return a dict."""
    try:
        # Remove ```json and ``` fences if the model adds them
        cleaned_text = re.sub(r"```json\s*|\s*```", "", text.strip())
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Fallback if model fails to generate valid JSON
        return {"error": "Failed to parse JSON", "raw_output": text}


    
def validation_pipeline(control_list, text_pages,file_uuid):
    tokenizer,model=load_model()
    for control in control_list:
        resp = validate_controls(text_pages[control['page']],control['control_id'],control['control_title'],control['control_desc'],tokenizer,model)
        print(resp)

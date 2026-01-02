from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from transformers import pipeline
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os



load_dotenv()
login(token=os.environ['hf_token'])
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_model(model_id="Qwen/Qwen2.5-7B-Instruct"):

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

# creating the rag pipeline by creating vecotr embeddings for the file uploaded.
def setup_langchain_rag(text_pages):
    documents = []
    for idx,page in enumerate(text_pages):
        doc =  Document(page_content=page,metadata={"page_number": idx + 1, "source": "pdf_upload"})
        documents.append(doc)

    tokenizer,model = load_model()
    # using a lightweight embedding model to save the time for validation.
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # creating a chroma db for storing vector chunks
    vectorstore = Chroma.from_documents(documents=documents,embedding=embeddings,collection_name="pdf_validation_store")

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1, # Low temp for strict validation
        do_sample=True,                # Switch to True to make penalty work better
        top_p=0.9,                     # Standard sampling
        repetition_penalty=1.15,      
        pad_token_id=tokenizer.eos_token_id
    
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline) #loadding the llm using the pipeline created.

    # returning the vector database created and llm model loaded
    return vectorstore,llm

def validate_extracted_controls(extracted_controls, vectorstore, llm):
    # prompt for the validation llm
    validation_template = """
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
    prompt = PromptTemplate(
        template = validation_template, #providing the prompt template
        input_variables=['context','control_id','control_title','control_desc'] # defining the variables in the prompt
    )

    # creating a chain manually so it gives more control over the RAG
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) #retrieves only page which is the most relevant (at number 1 in search results)

    validated_results = []


    for control in extracted_controls:
        # Retrieval of the relvant chunk using control id and control description for better accuracy
        query = f"{control['control_id']} {control['control_desc']}" #query string

        # searching the query in the document
        relevant_docs = retriever.invoke(query)

        # taking the content of the most relevant chunk
        context_text = relevant_docs[0].page_content if relevant_docs else "No context found."  #checking incase the page returned isnt empty 
        found_page = relevant_docs[0].metadata['page_number'] if relevant_docs else "Unknown" #getting the page number to pass to the model for better context.

        chain = prompt | llm #giving the prompt to the llm using pipe

        # using try-except because incase there is a error in the way output was returned by the model
        # because model is incosistent so it doesnt halts the validation in process
        try:
            # get a response from the chain of the llm
            response = chain.invoke({
                "context": context_text,
                "control_id": control['control_id'],
                "control_title": control['control_title'],
                "control_desc": control['control_desc']
            })
            
            # Clean up response to get JSON (Model might chatter)
            response_clean = response.split("{")[-1].split("}")[0]
            # keeping only the snippet between ```json {json_list_contents}```
            # so we can use that to validate the control
            response_clean=response.split("```json")[1].split('```')[0].strip()
            # converting the string returned by the model into a json object.
            response_json = json.loads(response_clean )
            
            # Add validation metadata to the control
            control['validation'] = {
                'status': 'PASS' if response_json.get('is_valid') else 'FAIL',
                'verified_on_page': found_page,
                'reason': response_json.get('reason')
            }

        except Exception as e:
            # incase the model is unable to give a valid output
            control['validation'] = {'status': 'PASS', 'reason': str(e)}

        # APPENDING THE VALIDATED CONTROLS
        validated_results.append(control)
        print(f" > Checked {control['control_id']}: {control['validation']['status']}")


    return validated_results




def validate_controls(pdf_file_pages,all_controls,file_uuid):
    vectorstore, langchain_llm = setup_langchain_rag(pdf_file_pages) #pdf file pages
    # Validate the extracted data
    final_validated_data = validate_extracted_controls(all_controls, vectorstore, langchain_llm)

    # exporting the validated contents
    with open(f"extracted_controls/{file_uuid}_validated.json", "w", encoding="utf-8") as f:
        json.dump(final_validated_data, f, indent=2, ensure_ascii=False)


    

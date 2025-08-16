import sys
from context_of_prompt import read_prompt_and_summarize
import logging
import re
import json
import pandas as pd
import os
from typing import List, Dict, Union
from typing_extensions import TypedDict
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
import vertexai
import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_google_vertexai").setLevel(logging.DEBUG)

# Initialize Vertex AI
PROJECT_ID = "vf-grp-aib-prd-cmr-cxi-lab"  # Use your actual project ID
LOCATION = "europe-west1"  # Use your preferred location
vertexai.init(project=PROJECT_ID, location=LOCATION)

class EvaluationState(TypedDict):
    summary: str
    assigned_l1: List[str]
    assigned_l2: Dict[str, List[str]]
    l1_evaluation: str
    l1_reasoning: str
    l2_evaluation: str
    l2_reasoning: str
    comment: str

def get_evaluation_prompt(generated_context):
    return ChatPromptTemplate.from_template(f"""
Tagging was performed on customer complaint summaries. The context below defines the topic hierarchy and rules for assigning Level 1 and Level 2 labels:

**********Beginning of context**************************
{generated_context}
*******End of context************

You are an AI evaluator tasked with assessing whether the assigned topic labels for a customer complaint are correct.

The complaint is: "{{summary}}"

Assigned Level 1 topics: {{assigned_l1}}
Assigned Level 2 topics: {{assigned_l2}}

**Evaluation Instructions:**
- Use the topic hierarchy and definitions from the context to evaluate the assigned labels.
- Summaries may have **multiple Level 1 topics**, and each Level 1 topic may have **multiple Level 2 subcategories**.
- **Level 1 Evaluation**: Determine if all assigned Level 1 topics in {{assigned_l1}} accurately reflect the complaint's issues and align with the Level 1 topic definitions (Email Problems, License Issues, Login Issues, System & Access Issues, Technical & Communication Issues, Other). Check if any relevant Level 1 topics are missing or if any assigned topics are irrelevant.
- **Level 2 Evaluation**: For each assigned Level 1 topic, verify if the corresponding Level 2 topics in {{assigned_l2}} are valid subcategories, align with the complaint's details, and match the Level 2 definitions. Ensure every Level 1 topic has relevant Level 2 topics; replace invalid labels with "Other" if specified in the context.
**Evaluation Outcomes**:
- **yes**: All labels for the level are accurate and complete.
- **partially yes**: At least one label is correct, but some are missing, extra, or slightly inaccurate.
- **no**: Labels are wrong, irrelevant, or violate the topic hierarchy.

**IMPORTANT**: Respond ONLY with a JSON object. No other text.

Example response:
{{
  "l1_evaluation": "yes",
  "l1_reasoning": "The assigned Level 1 topics correctly identify the main issues in the complaint.",
  "l2_evaluation": "partially yes", 
  "l2_reasoning": "Most Level 2 topics are appropriate but one is missing."
}}

Your actual response (use the exact same structure):
""")

def evaluate_complaint(state: EvaluationState, config: Dict) -> EvaluationState:
    llm_client = config.get("configurable", {}).get("llm_client")
    llm_type = config.get("configurable", {}).get("llm_type")
    summary = state["summary"]
    
    # Skip invalid summaries
    if not summary or str(summary).lower() == "nan":
        logger.warning(f"Skipping row with invalid summary: {summary}")
        return {
            "summary": summary,
            "assigned_l1": state["assigned_l1"],
            "assigned_l2": state["assigned_l2"],
            "l1_evaluation": "no",
            "l1_reasoning": "Invalid or missing summary (nan).",
            "l2_evaluation": "no",
            "l2_reasoning": "Invalid or missing summary (nan).",
            "comment": state["comment"]
        }
    
    prompt = config.get("evaluation_prompt").format(
        summary=summary,
        assigned_l1=state["assigned_l1"],
        assigned_l2=state["assigned_l2"]
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = [HumanMessage(content=prompt)]
            response = llm_client.invoke(messages)
            response_content = response.content
            logger.debug(f"Raw {llm_type} LLM response (attempt {attempt + 1}): {response_content} (length: {len(response_content)})")
            
            # Clean the response
            response_content = response_content.strip()
            if not response_content:
                logger.warning(f"Empty response from {llm_type} LLM on attempt {attempt + 1}")
                continue
            
            # Remove markdown and common Gemini output artifacts
            response_content = re.sub(r'^```json\s*', '', response_content, flags=re.MULTILINE)
            response_content = re.sub(r'\s*```$', '', response_content, flags=re.MULTILINE)
            response_content = re.sub(r'^```.*?\n', '', response_content, flags=re.MULTILINE)
            response_content = re.sub(r'\n```.*?$', '', response_content, flags=re.MULTILINE)
            response_content = response_content.strip()
            
            # Remove leading/trailing whitespace and newlines more aggressively
            response_content = response_content.strip('\n\r\t ')
            
            # Handle specific error patterns
            # Pattern 1: '\n  "l1_evaluation"' - just a key fragment
            if re.match(r'^[\n\r\s]*"[^"]*"[\n\r\s]*$', response_content):
                logger.warning(f"Response appears to be just a key fragment: '{response_content}'. Requesting retry.")
                continue
            
            # Pattern 2: ' and ends with ' - LLM interpreting prompt instructions literally
            # Be more aggressive in detecting this pattern
            problematic_phrases = [
                "and ends with",
                "starts with", 
                "ends with",
                "Your actual response",
                "use the exact same structure",
                "Example response"
            ]
            
            if any(phrase in response_content.lower() for phrase in problematic_phrases):
                logger.warning(f"Response contains instruction text: '{response_content[:100]}...'. Requesting retry.")
                continue
            
            # More careful handling of leading/trailing quotes - only remove if not part of valid JSON
            # Don't remove quotes if they're part of a valid JSON structure
            if not (response_content.startswith('{') and response_content.endswith('}')):
                # Only remove orphaned quotes at start/end if not valid JSON boundaries
                if response_content.startswith('"') and not response_content.startswith('{"'):
                    response_content = response_content[1:]
                if response_content.endswith('"') and not response_content.endswith('"}'):
                    response_content = response_content[:-1]
            
            # Fix common JSON issues
            response_content = response_content.replace("'", '"')
            response_content = re.sub(r',([}\]])', r'\1', response_content)  # Remove trailing commas
            response_content = re.sub(r'([{,]\s*)(\w+/\w+|\w+)(?=\s*:)', r'\1"\2"', response_content)  # Quote unquoted keys
            response_content = re.sub(r',\s*([}\]])', r'\1', response_content)  # Remove trailing commas before closing braces
            response_content = re.sub(r'"\s*,', r'",', response_content)  # Fix spaces before commas
            
            # Ensure proper JSON structure
            if not response_content.startswith('{'):
                response_content = '{' + response_content
            if not response_content.endswith('}'):
                response_content = response_content + '}'
            
            # Final cleanup: remove any remaining leading/trailing quotes outside braces
            response_content = re.sub(r'^"*\{', '{', response_content)
            response_content = re.sub(r'\}"*$', '}', response_content)
            logger.debug(f"Cleaned {llm_type} response (attempt {attempt + 1}): {response_content} (length: {len(response_content)})")
            
            # Validate JSON structure
            try:
                result = json.loads(response_content)
                required_fields = ["l1_evaluation", "l1_reasoning", "l2_evaluation", "l2_reasoning"]
                if not all(field in result for field in required_fields):
                    logger.warning(f"Missing required fields in JSON: {result}")
                    raise json.JSONDecodeError(f"Missing required JSON fields: {result}", response_content, 0)
                if result["l1_evaluation"] not in ["yes", "partially yes", "no"] or result["l2_evaluation"] not in ["yes", "partially yes", "no"]:
                    logger.warning(f"Invalid evaluation values in JSON: {result}")
                    raise json.JSONDecodeError(f"Invalid evaluation values: {result}", response_content, 0)
                break  # Successful parse
            except json.JSONDecodeError as e:
                logger.warning(f"JSONDecodeError on attempt {attempt + 1} for {llm_type} LLM, summary '{summary[:30]}...': {e}")
                logger.warning(f"Raw response causing error: '{response_content[:200]}...'")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to parse JSON after {max_retries} attempts for {llm_type} LLM")
                    logger.error(f"Final problematic response: '{response_content}'")
                    try:
                        # Fallback: Try parsing with ast.literal_eval
                        parsed = ast.literal_eval(response_content) if response_content else {}
                        if isinstance(parsed, dict):
                            result = {
                                "l1_evaluation": parsed.get("l1_evaluation", "no"),
                                "l1_reasoning": parsed.get("l1_reasoning", f"Failed to parse JSON after {max_retries} attempts: {str(e)}"),
                                "l2_evaluation": parsed.get("l2_evaluation", "no"),
                                "l2_reasoning": parsed.get("l2_reasoning", f"Failed to parse JSON after {max_retries} attempts: {str(e)}")
                            }
                            logger.info(f"Recovered partial JSON for {llm_type} LLM")
                        else:
                            raise json.JSONDecodeError("Invalid parsed object", response_content, 0)
                    except (json.JSONDecodeError, ValueError, SyntaxError) as e2:
                        logger.error(f"Failed to fix JSON for {llm_type} LLM: {e2}")
                        result = {
                            "l1_evaluation": "no",
                            "l1_reasoning": f"Failed to parse {llm_type} LLM response after {max_retries} attempts: {str(e2)}",
                            "l2_evaluation": "no",
                            "l2_reasoning": f"Failed to parse {llm_type} LLM response after {max_retries} attempts: {str(e2)}"
                        }
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1} for {llm_type} LLM, summary '{summary[:30]}...': {e}")
            if attempt == max_retries - 1:
                result = {
                    "l1_evaluation": "no",
                    "l1_reasoning": f"Processing error for {llm_type} LLM after {max_retries} attempts: {str(e)}",
                    "l2_evaluation": "no",
                    "l2_reasoning": f"Processing error for {llm_type} LLM after {max_retries} attempts: {str(e)}"
                }
                break
    
    state["l1_evaluation"] = result["l1_evaluation"]
    state["l1_reasoning"] = result["l1_reasoning"]
    state["l2_evaluation"] = result["l2_evaluation"]
    state["l2_reasoning"] = result["l2_reasoning"]
    return state

def create_workflow(evaluation_prompt):
    workflow = StateGraph(EvaluationState)
    workflow.add_node("evaluate_complaint", lambda state, config: evaluate_complaint(state, {**config, "evaluation_prompt": evaluation_prompt}))
    workflow.set_entry_point("evaluate_complaint")
    workflow.set_finish_point("evaluate_complaint")
    return workflow.compile()

def preprocess_excel(file_name: str) -> pd.DataFrame:
    file_path = os.path.join("CleanedDatasets", file_name)
    try:
        if not os.path.exists(file_path):
            logger.error(f"Excel file not found at '{file_path}'")
            return pd.DataFrame()
        df = pd.read_excel(file_path)
        logger.info(f"Successfully read Excel file '{file_path}'")
    except Exception as e:
        logger.error(f"Failed to read Excel file '{file_path}': {e}")
        return pd.DataFrame()
    
    # Verify required columns
    required_columns = ['summary', 'L1_L2_dict', 'L1: Actual Answer', 'L2:Actual Answer', 'comment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in Excel file: {missing_columns}")
        return pd.DataFrame()
    
    def clean_json_string(s: str) -> str:
        if not isinstance(s, str):
            return '{}'
        logger.debug(f"Original JSON string: {s}")
        s = re.sub(r'^\s*```json\s*|\s*```\s*$', '', s, flags=re.MULTILINE).strip()
        s = s.replace('{```json', '{').replace('```}', '}')
        s = s.replace('{\n{', '{').replace('}\n}', '}')
        s = s.replace("'", '"')
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        s = re.sub(r'([{,]\s*)(\w+/\w+|\w+)(?=\s*:)', r'\1"\2"', s)
        def clean_list_values(match):
            raw_items = match.group(1).split(',')
            cleaned_items = []
            for item in raw_items:
                item = item.strip()
                if not item:
                    continue
                item = re.sub(r'"*', '', item)
                item = item.replace('  ', ' ').strip()
                if ' ' in item:
                    item = ' '.join(word for word in item.split() if word)
                if item:
                    cleaned_items.append(f'"{item}"')
            result = ': [' + ','.join(cleaned_items) + ']'
            return result
        s = re.sub(r':\s*\[([^\]]*?)\]', clean_list_values, s)
        if 'positive' in s.lower():
            return '{"positive": ["positive"]}'
        pattern = r'("[^"]+": \["[^"]+"(?:,"[^"]+")*\])'
        match = re.search(pattern, s)
        if match:
            s = '{' + match.group(1) + '}'
        s = '{' + s + '}' if not s.startswith('{') else s
        try:
            parsed = json.loads(s)
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return json.dumps(parsed)
            except (ValueError, SyntaxError):
                pass
            return '{}'
    
    def parse_L1_L2_dict(x: Union[str, dict]) -> Dict[str, List[str]]:
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                cleaned = clean_json_string(x)
                result = json.loads(cleaned)
                return result if isinstance(result, dict) else {}
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse L1_L2_dict: {x}, error: {e}")
                return {}
        return {}
    
    def extract_level_1(x: Dict[str, List[str]]) -> List[str]:
        return list(x.keys())
    
    df['assigned_l2'] = df['L1_L2_dict'].apply(parse_L1_L2_dict)
    df['assigned_l1'] = df['assigned_l2'].apply(extract_level_1)
    df['summary'] = df['summary'].astype(str)
    df['comment'] = df['comment'].astype(str)
    df = df[['summary', 'assigned_l1', 'assigned_l2', 'L1: Actual Answer', 'L2:Actual Answer', 'comment']]
    return df

def process_row(row, llm_type: str, evaluation_prompt, project_id: str, location: str):
    llm_client = ChatVertexAI(model_name="gemini-1.5-flash", project=project_id, location=location, temperature=0, max_output_tokens=2048)
    app = create_workflow(evaluation_prompt)
    state = EvaluationState(
        summary=str(row["summary"]),
        assigned_l1=row["assigned_l1"],
        assigned_l2=row["assigned_l2"],
        l1_evaluation="",
        l1_reasoning="",
        l2_evaluation="",
        l2_reasoning="",
        comment=str(row["comment"])
    )
    try:
        start_llm = time.time()
        result = app.invoke(state, config={"configurable": {"llm_client": llm_client, "llm_type": llm_type}})
        end_llm = time.time()
        elapsed = end_llm - start_llm
        logger.info(f"{llm_type} LLM call took {elapsed:.3f} seconds for summary: {state['summary'][:30]}...")
        return {
            "index": row.name,
            "l1_evaluation": result["l1_evaluation"],
            "l1_reasoning": result["l1_reasoning"],
            "l2_evaluation": result["l2_evaluation"],
            "l2_reasoning": result["l2_reasoning"],
            "elapsed": elapsed
        }
    except Exception as e:
        logger.error(f"Error processing row for {llm_type} LLM, summary '{state['summary'][:30]}...': {e}")
        return {
            "index": row.name,
            "l1_evaluation": "no",
            "l1_reasoning": f"Processing error for {llm_type} LLM: {str(e)}",
            "l2_evaluation": "no",
            "l2_reasoning": f"Processing error for {llm_type} LLM: {str(e)}",
            "elapsed": 0.0
        }

def process_excel(df: pd.DataFrame, llm_type: str, evaluation_prompt, project_id: str, location: str) -> pd.DataFrame:
    l1_evaluations = [None] * len(df)
    l1_reasonings = [None] * len(df)
    l2_evaluations = [None] * len(df)
    l2_reasonings = [None] * len(df)
    total_llm_time = 0.0
    max_workers = min(10, len(df))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_row, row, llm_type, evaluation_prompt, project_id, location): row 
                         for _, row in df.iterrows()}
        for future in as_completed(future_to_row):
            result = future.result()
            index = result["index"]
            l1_evaluations[index] = result["l1_evaluation"]
            l1_reasonings[index] = result["l1_reasoning"]
            l2_evaluations[index] = result["l2_evaluation"]
            l2_reasonings[index] = result["l2_reasoning"]
            total_llm_time += result["elapsed"]
    df["L1_Evaluation"] = l1_evaluations
    df["L1_Reasoning"] = l1_reasonings
    df["L2_Evaluation"] = l2_evaluations
    df["L2_Reasoning"] = l2_reasonings
    df["L1_Match"] = df.apply(
        lambda x: x["L1: Actual Answer"].lower() == x["L1_Evaluation"].lower() or 
                  (x["L1: Actual Answer"].lower() == "y" and x["L1_Evaluation"].lower() == "yes") or 
                  (x["L1: Actual Answer"].lower() == "n" and x["L1_Evaluation"].lower() == "no"),
        axis=1
    )
    df["L2_Match"] = df.apply(
        lambda x: x["L2:Actual Answer"].lower() == x["L2_Evaluation"].lower() or 
                  (x["L2:Actual Answer"].lower() == "y" and x["L2_Evaluation"].lower() == "yes") or 
                  (x["L2:Actual Answer"].lower() == "n" and x["L2_Evaluation"].lower() == "no") or 
                  (x["L2:Actual Answer"].lower() == "yn" and x["L2_Evaluation"].lower() == "partially yes"),
        axis=1
    )
    print(f"Total {llm_type} LLM time for {len(df)} calls: {total_llm_time:.3f} seconds")
    return df

def evaluate_complaints(file_name: str, evaluation_prompt) -> pd.DataFrame:
    start_time = time.time()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    df = preprocess_excel(file_name)
    if df.empty:
        print("No data to process. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        result_df = process_excel(df.copy(), "vertex", evaluation_prompt, PROJECT_ID, LOCATION)
        result_df["Input_File"] = file_name
        result_df.to_excel("evaluated_complaints_gemini.xlsx", index=False)
    except Exception as e:
        logger.error(f"Error processing Gemini model: {e}")
        result_df = pd.DataFrame()
    total = len(df) if not df.empty else 0
    if not result_df.empty:
        l1_correct = len(result_df[result_df["L1_Match"] == True])
        l2_correct = len(result_df[result_df["L2_Match"] == True])
        l1_eval_counts = result_df["L1_Evaluation"].value_counts()
        l2_eval_counts = result_df["L2_Evaluation"].value_counts()
    else:
        l1_correct = l2_correct = 0
        l1_eval_counts = l2_eval_counts = pd.Series()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total complaints: {total}")
    print(f"\nExecution Time: {execution_time:.2f} seconds")
    print("\n=== Gemini LLM Metrics ===")
    print("Ground Truth Comparison:")
    print(f"L1 Evaluation Matches: {l1_correct} ({l1_correct/total*100:.2f}%)" if total > 0 else "L1 Evaluation Matches: 0 (0.00%)")
    print(f"L2 Evaluation Matches: {l2_correct} ({l2_correct/total*100:.2f}%)" if total > 0 else "L2 Evaluation Matches: 0 (0.00%)")
    print("Evaluation Distribution (L1):")
    for eval_type in ["yes", "partially yes", "no"]:
        count = l1_eval_counts.get(eval_type, 0)
        print(f"  {eval_type}: {count} ({count/total*100:.2f}%)" if total > 0 else f"  {eval_type}: 0 (0.00%)")
    print("Evaluation Distribution (L2):")
    for eval_type in ["yes", "partially yes", "no"]:
        count = l2_eval_counts.get(eval_type, 0)
        print(f"  {eval_type}: {count} ({count/total*100:.2f}%)" if total > 0 else f"  {eval_type}: 0 (0.00%)")
    return result_df

def run_evaluation(prompt_txt_file: str, excel_file: str) -> pd.DataFrame:
    generated_context = read_prompt_and_summarize(prompt_txt_file)
    evaluation_prompt = get_evaluation_prompt(generated_context)
    result_df = evaluate_complaints(excel_file, evaluation_prompt)
    return result_df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Evaluation2.py <prompt_txt_file> <excel_file>")
        print("Example: python Evaluation2.py tagging_prompt.txt CleanedDatasets/trial.xlsx")
        sys.exit(1)
    
    prompt_file = sys.argv[1]
    excel_file = sys.argv[2]
    
    print(f"Running evaluation with:")
    print(f"  Prompt file: {prompt_file}")
    print(f"  Excel file: {excel_file}")
    print("-" * 50)
    
    try:
        result_df = run_evaluation(prompt_file, excel_file)
        if not result_df.empty:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: evaluated_complaints_gemini.xlsx")
        else:
            print("No results generated. Check your input files and credentials.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

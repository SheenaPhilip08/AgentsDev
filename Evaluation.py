import sys
from context_of_prompt import read_prompt_and_summarize  # Import the summarization function
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
PROJECT_ID = "vf-grp-aib-prd-cmr-cxi-lab"
LOCATION = "europe-west1"
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
Tagging was performed on reviews. The below gives context into how the tags were allocated: 

**********Beginning of context**************************
{generated_context}
*******End of context************

You are an AI evaluator tasked with assessing whether the assigned topic labels assigned to every review are correct or not.

The complaint is: "{{summary}}"

Assigned Level 1 topics: {{assigned_l1}}
Assigned Level 2 topics: {{assigned_l2}}

Use the following topic hierarchy mentioned in the context to detrmine whether the tagging has been done correctly or not.  

**Evaluation Rules:**
- For Level 1: Evaluate if the assigned Level 1 topics (provided in {{assigned_l1}}) accurately reflect the complaint's main issue and are valid according to the Level 1 topic definitions and rules above. Remove any labels not in the allowed categories.
- For Level 2: Evaluate if the assigned Level 2 topics (provided in {{assigned_l2}}) are correct subcategories under the assigned Level 1 topics, align with the complaint's details, and are valid according to the Level 2 topic definitions and rules above. Ensure every Level 1 topic has a relevant Level 2 topic, and remove or replace invalid labels with "other" as specified.
- Evaluation outcomes:
  - yes: All labels (for the respective level) are accurate and complete.
  - partially yes: At least one label is correct, but some are missing, extra, or slightly inaccurate.
  - no: Labels are wrong, irrelevant, or violate the topic hierarchy and rules.
- Ensure the output is valid JSON with the exact structure shown below.

Output in JSON format (use double quotes and proper JSON syntax):
{{{{
  "l1_evaluation": "yes|partially yes|no",
  "l1_reasoning": "Explain why the Level 1 labels are accurate or not, based on the complaint and the topic definitions.",
  "l2_evaluation": "yes|partially yes|no",
  "l2_reasoning": "Explain why the Level 2 labels are accurate or not, based on the complaint and the topic definitions"
}}}}
""")

def evaluate_complaint(state: EvaluationState, config: Dict) -> EvaluationState:
    llm_client = config.get("configurable", {}).get("llm_client")
    llm_type = config.get("configurable", {}).get("llm_type")
    prompt = config.get("evaluation_prompt").format(
        summary=state["summary"],
        assigned_l1=state["assigned_l1"],
        assigned_l2=state["assigned_l2"]
    )
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm_client.invoke(messages)
        response_content = response.content
        response_content = re.sub(r'^```json\s*|\s*```$', '', response_content, flags=re.MULTILINE).strip()
        response_content = re.sub(r',([}\]])', r'\1', response_content)
        response_content = re.sub(r'[^\x20-\x7E]', '', response_content)
        logger.debug(f"Cleaned {llm_type} response: {response_content}")
        result = json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError for {llm_type} LLM, summary '{state['summary']}': {e}")
        logger.error(f"Raw response: {response_content}")
        try:
            response_content = response_content.replace("'", '"')
            response_content = re.sub(r'([{,]\s*)(\w+/\w+|\w+)(?=\s*:)', r'\1"\2"', response_content)
            result = json.loads(response_content)
            logger.info(f"Successfully parsed after fixing JSON for {llm_type} LLM")
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to fix JSON for {llm_type} LLM: {e2}")
            result = {
                "l1_evaluation": "no",
                "l1_reasoning": f"Failed to parse {llm_type} LLM response after cleaning: {str(e2)}",
                "l2_evaluation": "no",
                "l2_reasoning": f"Failed to parse {llm_type} LLM response after cleaning: {str(e2)}"
            }
    except Exception as e:
        logger.error(f"Error for {llm_type} LLM, summary '{state['summary']}': {e}")
        result = {
            "l1_evaluation": "no",
            "l1_reasoning": f"Processing error for {llm_type} LLM: {str(e)}",
            "l2_evaluation": "no",
            "l2_reasoning": f"Processing error for {llm_type} LLM: {str(e)}"
        }
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
    def clean_json_string(s: str) -> str:
        """Clean JSON string by removing markdown code fences, fixing common issues, and handling edge cases."""
        if not isinstance(s, str):
            return '{}'
        
        logger.debug(f"Original JSON string: {s}")
        
        # Remove outer quotes if present (handles quoted JSON strings)
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        elif s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        
        # Fix escaped quotes from JSON-within-JSON
        s = s.replace('\\"', '"').replace("\\'", "'")
        
        # Remove markdown code fences and nested braces
        s = re.sub(r'^\s*```json\s*', '', s, flags=re.MULTILINE)
        s = re.sub(r'\s*```\s*$', '', s, flags=re.MULTILINE)
        s = s.replace('{```json', '{').replace('```}', '}')
        s = s.replace('{\n{', '{').replace('}\n}', '}')
        logger.debug(f"After removing markdown and braces: {s}")
        
        # Fix double quotes issue: ""key"" -> "key"
        s = re.sub(r'""([^"]*?)""', r'"\1"', s)
        logger.debug(f"After fixing double quotes: {s}")
        
        # Replace single quotes with double quotes
        s = s.replace("'", '"')
        logger.debug(f"After replacing quotes: {s}")
        
        # Fix trailing commas
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        logger.debug(f"After fixing trailing commas: {s}")
        
        # Quote unquoted keys (e.g., hardware/software -> "hardware/software")
        s = re.sub(r'([{,]\s*)(\w+[^":\s]*?)(?=\s*:)', r'\1"\2"', s)
        logger.debug(f"After quoting keys: {s}")
        
        # Try to parse the cleaned JSON
        try:
            parsed = json.loads(s)
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            logger.debug(f"First JSON parse failed: {e}")
            
            # Fallback: try ast.literal_eval
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return json.dumps(parsed)
            except (ValueError, SyntaxError) as e2:
                logger.debug(f"AST parse also failed: {e2}")
                
            # Final fallback: return empty dict
            logger.warning(f"Could not parse JSON string: {s}")
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
                return {}
        return {}
    def extract_level_1(x: Dict[str, List[str]]) -> List[str]:
        return list(x.keys())
    df['assigned_l2'] = df['L1_L2_dict'].apply(parse_L1_L2_dict)
    df['assigned_l1'] = df['assigned_l2'].apply(extract_level_1)
    df['summary'] = df['summary'].astype(str)
    df['comment'] = df['comment'].astype(str)
    required_columns = ['summary', 'assigned_l1', 'assigned_l2', 'L1: Actual Answer', 'L2:Actual Answer', 'comment']
    df = df[[col for col in required_columns if col in df.columns]]
    return df

def process_row(row, llm_type: str, evaluation_prompt, project_id: str, location: str):
    # Create a new LLM client for each thread to ensure thread safety
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
        logger.error(f"Error processing row for {llm_type} LLM, summary '{state['summary']}': {e}")
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
    
    # Use ThreadPoolExecutor to parallelize row processing
    max_workers = min(10, len(df))  # Limit to 10 workers or number of rows
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
    
    # Adjust column names to match the sample data for matching
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
        # Add the input file name as a new column
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

def generate_improvement_suggestions(result_df: pd.DataFrame, generated_context: str) -> str:
    """
    Generate overall topic structure improvement suggestions based on all evaluations
    """
    if result_df.empty:
        return "No data available for improvement analysis."
    
    # Analyze patterns in the evaluation results
    l1_issues = result_df[result_df["L1_Evaluation"] != "yes"]
    l2_issues = result_df[result_df["L2_Evaluation"] != "yes"]
    
    # Prepare summary data for the LLM
    total_complaints = len(result_df)
    l1_accuracy = len(result_df[result_df["L1_Evaluation"] == "yes"]) / total_complaints * 100
    l2_accuracy = len(result_df[result_df["L2_Evaluation"] == "yes"]) / total_complaints * 100
    
    # Sample problematic cases
    problem_cases = []
    for _, row in l1_issues.head(5).iterrows():
        problem_cases.append({
            "summary": row["summary"][:200] + "..." if len(row["summary"]) > 200 else row["summary"],
            "assigned_l1": row["assigned_l1"],
            "assigned_l2": row["assigned_l2"],
            "l1_reasoning": row["L1_Reasoning"],
            "l2_reasoning": row["L2_Reasoning"]
        })
    
    improvement_prompt = f"""
You are an expert in topic classification systems. Based on the evaluation results of {total_complaints} customer complaints, provide recommendations to improve the overall topic hierarchy structure.

**Current Performance:**
- Level 1 Accuracy: {l1_accuracy:.1f}%
- Level 2 Accuracy: {l2_accuracy:.1f}%

**Current Topic Structure:**
{generated_context}

**Sample Problematic Cases:**
{json.dumps(problem_cases, indent=2)}

**Analysis Request:**
Based on the performance data and problematic cases, provide specific recommendations to improve the topic classification system:

1. **Missing Categories**: What new Level 1 or Level 2 topics should be added?
2. **Overlapping Categories**: Which existing categories cause confusion and how should they be refined?
3. **Definition Improvements**: How can category definitions be clarified?
4. **Rule Modifications**: What tagging rules should be adjusted?
5. **Structural Changes**: Should any categories be merged, split, or reorganized?

Provide actionable recommendations that would improve classification accuracy and consistency.

**Output Format:**
Provide a clear, structured analysis with specific recommendations for improving the topic hierarchy.
"""

    try:
        llm_client = ChatVertexAI(
            model_name="gemini-1.5-flash", 
            project=PROJECT_ID, 
            location=LOCATION, 
            temperature=0.3,  # Slightly higher for more creative suggestions
            max_output_tokens=3000
        )
        
        response = llm_client.invoke([HumanMessage(content=improvement_prompt)])
        return response.content
        
    except Exception as e:
        logger.error(f"Failed to generate improvement suggestions: {e}")
        return f"Error generating improvement suggestions: {str(e)}"

def run_evaluation_with_improvements(prompt_txt_file: str, excel_file: str) -> pd.DataFrame:
    """
    Run the complete evaluation process and generate improvement suggestions
    """
    print("Starting evaluation with improvement analysis...")
    
    # Generate context from prompt file
    generated_context = read_prompt_and_summarize(prompt_txt_file)
    evaluation_prompt = get_evaluation_prompt(generated_context)
    
    # Run the main evaluation
    result_df = evaluate_complaints(excel_file, evaluation_prompt)
    
    if not result_df.empty:
        print("\n" + "="*80)
        print("TOPIC STRUCTURE IMPROVEMENT SUGGESTIONS")
        print("="*80)
        
        # Generate improvement suggestions based on results
        improvements = generate_improvement_suggestions(result_df, generated_context)
        print(improvements)
        
        # Save improvements to file
        with open("topic_improvement_suggestions.txt", "w") as f:
            f.write("TOPIC STRUCTURE IMPROVEMENT SUGGESTIONS\n")
            f.write("="*50 + "\n\n")
            f.write(improvements)
        
        print(f"\nImprovement suggestions saved to: topic_improvement_suggestions.txt")
    
    return result_df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Evaluation.py <prompt_txt_file> <excel_file>")
        print("Example: python Evaluation.py tagging_prompt.txt CleanedDatasets/trial.xlsx")
        sys.exit(1)
    
    prompt_file = sys.argv[1]
    excel_file = sys.argv[2]
    
    print(f"Running evaluation with improvement analysis:")
    print(f"  Prompt file: {prompt_file}")
    print(f"  Excel file: {excel_file}")
    print("-" * 50)
    
    try:
        result_df = run_evaluation_with_improvements(prompt_file, excel_file)
        if not result_df.empty:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: evaluated_complaints_gemini.xlsx")
            print(f"Improvement suggestions saved to: topic_improvement_suggestions.txt")
        else:
            print("No results generated. Check your input files and credentials.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

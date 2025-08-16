import logging
import re
import os
import sys
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
import vertexai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("langchain_google_vertexai").setLevel(logging.DEBUG)

# Initialize Vertex AI for Gemini
PROJECT_ID = "vf-grp-aib-prd-cmr-cxi-lab"
LOCATION = "europe-west1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize Gemini 2.5 Flash model
gemini_model = ChatVertexAI(
    model_name="gemini-1.5-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
    max_tokens=4000  # Increased to handle detailed rule extraction
)

def read_prompt_and_summarize(prompt_file: str) -> str:
    """
    Reads a prompt file and uses Gemini 2.5 Flash to extract:
    - Task background
    - Level 1 topic definitions
    - Level 2 topics grouped under Level 1 with definitions or rules
    - General and conditional tagging rules

    Args:
        prompt_file (str): Path to the prompt file

    Returns:
        str: Structured markdown summary
    """
    prompt_path = os.path.join(os.getcwd(), prompt_file)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            raw_prompt = f.read()
            logger.info(f"Successfully read prompt file: {prompt_path}")
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        return "**Error**: Prompt file not found."
    except Exception as e:
        logger.error(f"Failed to read prompt file: {str(e)}")
        return f"**Error**: {str(e)}"

    # Clean content (no truncation)
    cleaned_prompt = re.sub(r'[^\x20-\x7E\n]', '', raw_prompt)
    logger.info(f"Prompt content length: {len(cleaned_prompt)} characters")

    summarization_prompt = f"""
You are given a prompt that defines a hierarchical tagging or classification system for analyzing text data, such as customer feedback, survey responses, or complaints. The system includes Level 1 (high-level) topics, Level 2 (subcategory) topics, and rules for applying labels, including general and conditional rules.

Your task is to extract and summarize all key components of the prompt in a structured format, ensuring Level 2 topics are correctly grouped under their corresponding Level 1 topics with their definitions or rules, and all tagging rules are fully captured, including every conditional rule provided.

---

**Instructions**:

1. **Background**:
   - Summarize the purpose or goal of the tagging/classification task in 1–2 sentences.
   - If no explicit purpose is stated, infer it based on the context and describe it concisely.

2. **Level 1 Topics**:
   - List each Level 1 topic explicitly mentioned in the prompt.
   - Include any definition, description, or context provided for each Level 1 topic.
   - If no definition is provided, state "Not provided".
   - Format: `- topic: description`

3. **Level 2 Topics**:
   - For each Level 1 topic, list all associated Level 2 subtopics as defined in the prompt, including those specified in rules like 'If the input labels include [Level 1] then choose...'.
   - Ensure Level 2 topics are grouped under their correct Level 1 topic based on the prompt’s structure.
   - Include any definitions, descriptions, or specific tagging rules for each Level 2 subtopic.
   - If no description or rule is provided, state "Not provided".
   - If a Level 1 topic has no associated Level 2 topics, state "No Level 2 topics defined".
   - Format:
     ```
     - Level 1 topic:
         - Level 2 subtopic: description or tagging rule
         - Level 2 subtopic: description or tagging rule
     ```

4. **General Tagging Rules**:
   - Extract all general rules or principles that govern how tags or labels should be applied or excluded, including any instructions about handling positive feedback, vague feedback, or invalid labels.
   - Examples: "Only tag negative feedback", "Use a default category for vague input", or "Remove positive mentions".
   - If none are provided, state "None provided".
   - Format: `- rule`

5. **Conditional Labeling Rules**:
   - Extract every rule that specifies conditions for using or avoiding specific labels based on the content or phrasing of the text, including all rules listed under 'Before you choose from the labels, make sure to follow the below rules'.
   - Map each rule to the specific label or topic it applies to, indicating whether it affects Level 1 or Level 2 labels.
   - Include all conditional rules, even if they overlap with Level 2 topic definitions, to ensure completeness.
   - Examples: "Use 'X' only if Y is mentioned", "Avoid 'Z' if the issue is unrelated to Q".
   - If none are provided, state "None provided".
   - Format: `- label (Level 1 or Level 2): rule or condition`

---

**Prompt Content**:
{cleaned_prompt}

---

**Output Format**:
**Background**:
<summary>

**Level 1 Topics**:
- topic: description

**Level 2 Topics**:
- Level 1 topic:
    - Level 2 subtopic: description or tagging rule
    - Level 2 subtopic: description or tagging rule

**General Tagging Rules**:
- rule 1
- rule 2

**Conditional Labeling Rules**:
- label (Level 1 or Level 2): rule or condition
- label (Level 1 or Level 2): rule or condition
"""

    try:
        response = gemini_model.invoke([HumanMessage(content=summarization_prompt)])
        result = response.content.strip()
        logger.info("Gemini 2.5 Flash successfully generated summary.")
        return result
    except Exception as e:
        logger.error(f"Summarization failed with Gemini: {str(e)}")
        return f"**Error**: Failed to summarize prompt with Gemini. Reason: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 'context of prompt' <prompt_txt_file>")
        sys.exit(1)
    prompt_file = sys.argv[1]
    summary = read_prompt_and_summarize(prompt_file)
    print(f"\n{'='*80}\nSummarizing {prompt_file}:\n{'='*80}")
    print(summary)
    print("\n" + "-"*80)

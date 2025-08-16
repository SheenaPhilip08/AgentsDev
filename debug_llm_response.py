#!/usr/bin/env python3
"""
Debug script to test LLM response and identify the issue
"""

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
import vertexai
import json

# Initialize Vertex AI (you'll need to update these)
PROJECT_ID = "vf-grp-aib-prd-cmr-cxi-lab"
LOCATION = "europe-west1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Test the LLM with a simple prompt
llm_client = ChatVertexAI(
    model_name="gemini-1.5-flash", 
    project=PROJECT_ID, 
    location=LOCATION, 
    temperature=0, 
    max_output_tokens=2048
)

test_prompt = """
You are evaluating topic labels for a customer complaint.

The complaint is: "The customer called about password reset issues and cannot access their email."

Assigned Level 1 topics: ["login issues"]
Assigned Level 2 topics: {"login issues": ["Password resets & credentials"]}

Evaluate if these labels are correct.

**IMPORTANT**: Respond ONLY with a JSON object. No other text.

Example response:
{
  "l1_evaluation": "yes",
  "l1_reasoning": "The assigned Level 1 topics correctly identify the main issues in the complaint.",
  "l2_evaluation": "partially yes", 
  "l2_reasoning": "Most Level 2 topics are appropriate but one is missing."
}

Your actual response (use the exact same structure):
"""

print("Testing LLM response...")
print("=" * 50)

try:
    response = llm_client.invoke([HumanMessage(content=test_prompt)])
    response_content = response.content
    
    print(f"Raw response: '{response_content}'")
    print(f"Response length: {len(response_content)}")
    print(f"Response type: {type(response_content)}")
    print()
    
    # Try to parse as JSON
    try:
        parsed = json.loads(response_content.strip())
        print("SUCCESS: Valid JSON parsed!")
        print(f"Parsed content: {parsed}")
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed: {e}")
        print()
        
        # Show character codes for debugging
        print("First 50 characters as ASCII codes:")
        for i, char in enumerate(response_content[:50]):
            print(f"  {i}: '{char}' (ASCII {ord(char)})")
            
except Exception as e:
    print(f"LLM call failed: {e}")
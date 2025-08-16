# Customer Complaint Topic Evaluation System

This system evaluates the accuracy of AI-generated topic classifications for customer support complaints using Google's Vertex AI (Gemini) models.

## Overview

The system consists of three main components:

1. **`context_of_prompt.py`** - Extracts and summarizes tagging rules from prompt files
2. **`Evaluation2.py`** - Evaluates topic assignments against ground truth data
3. **Data files** - Excel files containing complaint summaries and their topic classifications

## How It Works

### 1. Topic Classification System

The system uses a **two-level hierarchical classification**:

- **Level 1**: High-level categories (e.g., "Login Issues", "Technical & Communication Issues")
- **Level 2**: Specific subcategories under each Level 1 topic

### 2. Evaluation Process

1. **Context Extraction**: The system reads a prompt file defining the topic hierarchy and rules
2. **Data Processing**: It loads Excel data containing:
   - Customer complaint summaries
   - Assigned Level 1 topics (from AI classification)
   - Assigned Level 2 topics (from AI classification)
   - Ground truth labels (`L1: Actual Answer`, `L2:Actual Answer`)

3. **LLM Evaluation**: For each complaint, an LLM evaluates whether the assigned topics are correct based on:
   - The complaint summary
   - The topic definitions from the prompt
   - The tagging rules

4. **Scoring**: The system compares LLM evaluations with ground truth to calculate accuracy metrics

## File Structure

```
/workspace/
├── Evaluation2.py              # Main evaluation script
├── context_of_prompt.py        # Prompt processing utility
├── tagging_prompt.txt          # Topic hierarchy and rules definition
├── CleanedDatasets/
│   └── trial.xlsx             # Sample complaint data
├── create_sample_data.py       # Script to generate sample data
└── README.md                   # This documentation
```

## Excel Data Format

Your Excel file should contain these columns:

| Column | Description |
|--------|-------------|
| `survey_id` | Unique identifier for each complaint |
| `summary` | Customer complaint summary text |
| `level_1` | JSON string of assigned Level 1 topics |
| `L1_L2_dict` | JSON string mapping L1 topics to L2 subtopics |
| `L1: Actual Answer` | Ground truth for L1 evaluation (Y/N) |
| `L2:Actual Answer` | Ground truth for L2 evaluation (Y/N) |

### Example Data Row:
```
survey_id: "2c04b3ca-cd4a-4bd2-a13e-a29466493838"
summary: "The customer called the One IT Service Desk for assistance..."
level_1: ["login issues"]
L1_L2_dict: {"login issues": ["Password resets & credentials", "Account login/logout problems"]}
L1: Actual Answer: "Y"
L2:Actual Answer: "Y"
```

## Usage

### Prerequisites

1. Install required Python packages:
```bash
pip install pandas openpyxl langchain-google-vertexai langgraph
```

2. Set up Google Cloud credentials for Vertex AI access

3. Update `PROJECT_ID` and `LOCATION` in `Evaluation2.py` with your Google Cloud project details

### Running the Evaluation

```bash
python3 Evaluation2.py <prompt_file> <excel_file>
```

**Example:**
```bash
python3 Evaluation2.py tagging_prompt.txt CleanedDatasets/trial.xlsx
```

### Creating Sample Data

To generate sample data for testing:
```bash
python3 create_sample_data.py
```

## Output

The system generates:

1. **Console Output**: Real-time progress and summary statistics
2. **Excel File**: `evaluated_complaints_gemini.xlsx` with evaluation results

### Output Columns Added:
- `L1_Evaluation`: LLM's evaluation of Level 1 topics ("yes", "partially yes", "no")
- `L1_Reasoning`: Explanation for L1 evaluation
- `L2_Evaluation`: LLM's evaluation of Level 2 topics
- `L2_Reasoning`: Explanation for L2 evaluation
- `L1_Match`: Whether LLM evaluation matches ground truth
- `L2_Match`: Whether LLM evaluation matches ground truth

### Sample Output:
```
Total complaints: 5

Execution Time: 15.23 seconds

=== Gemini LLM Metrics ===
Ground Truth Comparison:
L1 Evaluation Matches: 4 (80.00%)
L2 Evaluation Matches: 5 (100.00%)

Evaluation Distribution (L1):
  yes: 4 (80.00%)
  partially yes: 1 (20.00%)
  no: 0 (0.00%)
```

## Customizing the System

### 1. Modify Topic Hierarchy

Edit `tagging_prompt.txt` to:
- Add new Level 1 or Level 2 topics
- Update topic definitions
- Modify tagging rules

### 2. Adjust Evaluation Criteria

In `Evaluation2.py`, modify the `get_evaluation_prompt()` function to change:
- Evaluation instructions
- Scoring criteria
- Output format requirements

### 3. Change LLM Model

Update the model configuration in `Evaluation2.py`:
```python
# Current: Gemini 1.5 Flash
llm_client = ChatVertexAI(model_name="gemini-1.5-flash", ...)

# Alternative: Gemini Pro
llm_client = ChatVertexAI(model_name="gemini-1.5-pro", ...)
```

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**: Install all required packages using pip
2. **Authentication Errors**: Ensure Google Cloud credentials are properly configured
3. **Empty Results**: Check that your Excel file has the correct column names and data format
4. **JSON Parsing Errors**: Verify that `level_1` and `L1_L2_dict` columns contain valid JSON

### Debug Mode:

Enable verbose logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Parallel Processing**: The system uses ThreadPoolExecutor for concurrent LLM calls
- **Rate Limiting**: Adjust `max_workers` in `process_excel()` to control API call rate
- **Batch Size**: Process large datasets in smaller batches if needed

## Security Notes

- Store Google Cloud credentials securely
- Avoid committing actual project IDs to version control
- Use environment variables for sensitive configuration
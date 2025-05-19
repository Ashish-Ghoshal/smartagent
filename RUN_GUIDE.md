# RUN_GUIDE.md

This is a beginner-friendly, step-by-step guide for running the SmartAgent project using Visual Studio Code (VS Code).

## 1. Requirements

- Python 3.10+ (already installed)
- Visual Studio Code (VS Code)
- OpenAI API key (get from: https://platform.openai.com/account/api-keys)

## 2. Open the Project in VS Code

1. Open VS Code
2. Click on "File" → "Open Folder..."
3. Navigate to and select the folder: `smartagent_v2.6.1_cleaned_structure`

You should now see folders like `src`, `data`, and `tests` in the Explorer panel.

## 3. Create and Activate a Virtual Environment

Open the integrated terminal (Terminal → New Terminal), then run:

### On Windows:
```
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` in your terminal prompt.

## 4. Install Project Dependencies

If `requirements.txt` exists:
```
pip install -r requirements.txt
```

If it doesn't, install these manually:
```
pip install langchain openai pandas scikit-learn statsmodels pytest
```

Optional tools you may need:
```
pip install matplotlib seaborn
```

## 5. Configure Your OpenAI API Key

1. Navigate to:
   `src/react_agent/config.py`

2. Open the file and replace:
```
OPENAI_API_KEY = "your_openai_api_key_here"
```
with your actual key:
```
OPENAI_API_KEY = "sk-abc123..."
```

3. Save the file.

## 6. Add a CSV File

Place a CSV file into the folder:
```
data/tables/
```
Example: `sales.csv`

This is the file the SmartAgent will analyze.

## 7. Run the SmartAgent in VS Code

Open the VS Code terminal and launch Python:

```
python
```

Then enter the following code:

```python
from react_agent.agents.smart_controller_memory import smart_agent_with_memory
smart_agent_with_memory.run("Generate an EDA report for sales.csv")
```

You should see a structured response from the agent.

## 8. Run a Follow-Up Command

While still in the Python shell, you can run:
```python
smart_agent_with_memory.run("Now cluster it")
```

The agent will remember the earlier command and build on it.

## 9. Running Tests (Optional)

In your terminal, run:
```
pytest tests/
```

---

This guide ensures you can set up and run the SmartAgent project entirely from within VS Code with minimal prior experience.

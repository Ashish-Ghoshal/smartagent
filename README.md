# SmartAgent

SmartAgent is a modular, LLM-powered assistant designed to help users perform data analysis tasks on CSV files using natural language. The system routes queries to the appropriate tool for tasks like EDA, model selection, clustering, anomaly detection, and dimensionality reduction.

## Features

- EDA report generation
- LLM-guided model selection
- ML model training and testing
- Unsupervised clustering with silhouette scoring
- Dimensionality reduction with PCA
- Outlier detection with Isolation Forest and LOF
- Cluster summarization using LLMs
- Conversational memory support for follow-up queries
- Beginner-friendly setup guide (`RUN_GUIDE.md`)
- Full test suite

## Folder Structure

```
.
├── agents/                          # Legacy (can be removed)
├── data/
│   └── tables/                      # Place CSV files here
├── rag/                             # RAG components (for document-based tools)
├── src/
│   └── react_agent/
│       ├── agents/                  # Analyst + memory controller agents
│       ├── config.py                # OpenAI API key setup
│       ├── configuration.py
│       ├── graph.py
│       ├── logger.py
│       ├── new_tools/              # All task-specific tools live here
│       ├── prompts.py
│       ├── state.py
│       └── tools.py
├── static/
├── tests/
│   ├── integration_tests/          # Functional test coverage
│   └── unit_tests/
├── RUN_GUIDE.md                    # Beginner-friendly instructions
├── NOTES.md                        # Internal roadmap and notes
├── README.md
└── requirements.txt
```

## Flow of Execution

1. The user enters a query (e.g., "Generate an EDA report for sales.csv")
2. The memory-enabled `smart_controller_memory.py` receives the query
3. The agent selects the appropriate tool from `analyst_agent.py`
4. The selected tool is executed using LangChain
5. The result is returned and optionally stored in memory
6. Follow-up queries can build on prior results

## Setup Instructions

1. Install Python 3.10+
2. Clone the project and install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your CSV files to the `data/tables/` folder
4. Paste your OpenAI API key into `src/react_agent/config.py`
5. Open a Python shell or script and run:
```python
from react_agent.agents.smart_controller_memory import smart_agent_with_memory
smart_agent_with_memory.run("Generate an EDA report for sales.csv")
```

## Running Tests

To run tests:
```bash
pytest tests/
```

## Limitations

- Assumes CSVs are placed manually in `data/tables/`
- OpenAI key must be manually configured
- Memory is session-based only (not persistent)
- Not optimized for very large datasets or real-time interactivity

## Future Enhancements

SmartAgent is designed to be modular and scalable. Below is a breakdown of potential future improvements and how they would enhance the overall project.

### 1. Time Series Support
- Implement time series decomposition using STL or seasonal-trend modeling
- Forecast future values using ARIMA, Prophet, or exponential smoothing
- Detect time-based anomalies using z-scores, rolling averages, and trend residuals

### 2. Persistent Memory
- Add memory storage using Redis, SQLite, or JSON logs
- Support recall across sessions (like remembering the last used file or model)
- Enable more intelligent multi-turn conversations

### 3. File Upload Interface
- Add a simple web UI using Streamlit or Flask
- Allow drag-and-drop CSV uploads
- Display preview of uploaded data, and allow interaction with SmartAgent

### 4. Cloud Storage Integration
- Connect to Amazon S3, Google Cloud Storage, or Azure Blob Storage
- Allow SmartAgent to pull data from cloud buckets
- Add config-based access keys for enterprise use cases

### 5. LangGraph or Multi-Agent Support
- Use LangGraph or AgentExecutor to support complex workflows
- Split responsibilities across multiple agents (e.g., one for ML, one for visualizations)
- Enable task chaining and goal-driven prompting

### 6. Scalability Enhancements
- Containerize the application using Docker
- Add a RESTful API layer for integration with external systems
- Host on cloud platforms (e.g., AWS, Heroku, or GCP) for public access

These improvements aim to transition SmartAgent from a prototype to a production-ready tool, capable of powering real-world applications in business analytics, education, and data science workflows.
## Author

Developed by [Your Name] as a demonstration of applied agentic AI systems with LangChain.

# Bank Marketing AI Analytics Agent
An autonomous data agent that predicts bank deposit subscriptions and provides AI-driven business insights.

## Project Highlights
- **Leakage-Free ML:** Engineered a pipeline that identifies and removes predictive bias (e.g., call duration) to ensure real-world reliability.
- **Agentic Reasoning:** Integrated Gemini 3 Flash to translate complex model coefficients into executive business strategies.
- **Interactive Dashboard:** Built a Streamlit interface for seamless CSV data processing and visualization.

## Tech Stack
- **AI/LLM:** Google Gemini 2.5 (v1 API)
- **ML Framework:** Scikit-Learn (Pipelines, Logistic Regression)
- **Data:** Pandas, Numpy
- **UI:** Streamlit

## How to Run
1. Clone this repo.
2. Install libraries: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to a `.env` file.
4. Run: `streamlit run app.py`
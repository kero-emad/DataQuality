# DataQuality 🔍

[![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/AI-Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, interactive dashboard for **Data Profiling, Cleaning, and AI-assisted Analysis**. Built with Streamlit, this tool simplifies the process of transforming raw datasets into high-quality, analysis-ready data.

---

## 🚀 Features Breakdown

### 📊 1. Data Profiling & Inspection
*   **View Dataset Info**: Quick access to column names, non-null counts, and data types to understand your data structure.
*   **Describe Dataset**: Instant statistical summary (mean, std, min, max, etc.) for all numerical columns.
*   **View Dataset**: Interactive data explorer with a slider to select and view a specific number of rows from the head or tail.
*   **Before Download**: A final sanity check showing dataset size, missing cell percentages, and duplicate row counts before exporting.

### 🛠️ 2. Data Cleaning
*   **Handle Invalid Numeric Data**: Automatically detects non-numeric strings in columns that should be numbers and offers to replace them with `NaN`.
*   **Handle Missing Values**: 
    *   **Visual Analysis**: Heatmaps to identify patterns of missing data.
    *   **Smart Imputation**: Fill gaps using Mean, Median, or Mode, or choose to delete rows with missing values.
*   **Handle Duplicate Rows**: One-click detection and removal of identical rows to ensure data integrity.
*   **Handle Outliers**: 
    *   **Box Plot Visualization**: View data spread and identify extreme values.
    *   **Actionable Fixes**: Choose between **Dropping** outliers or **Clipping** them to the Upper/Lower bounds.

### 🔄 3. Data Transformation
*   **Handle Column Names**: Rename any column directly from the UI for better clarity.
*   **Data Type Convert**: Safely cast columns between `int`, `float`, `string`, and `datetime`.
*   **Drop Columns**: Clean up your workspace by removing redundant or unnecessary features.
*   **Handle Classification Columns**: Convert categorical text labels into numerical mappings for Machine Learning readiness.

### 📈 4. Advanced Analytics & Vis
*   **Visualization**: Generate high-quality Histograms for numerical trends and Pie Charts for categorical distributions.
*   **Correlation**: Heatmap visualization of the relationship between all numerical variables.
*   **Interactions**: Scatter plot explorer to visualize how two different variables interact with each other.

### 📋 5. System & AI Integration
*   **View Log**: A persistent session log that tracks every operation you perform during your data cleaning session.
*   **Chat (Dataset Context)**: A built-in assistant that answers specific questions about your data (e.g., "Which columns have missing values?").
*   **Ollama (AI Assistant)**: Advanced integration with Llama 3.2 for complex conversational analysis and insights generation.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Large Language Model**: [LangChain Ollama](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/)

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.13 or higher
- [Ollama](https://ollama.com/) (installed and running with `llama3.2` model for chat features)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kero-emad/DataQuality.git
   cd DataQuality
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn langchain-ollama
   ```

### Running the App

```bash
streamlit run p.py
```


## 🤝 Contributing

Contributions are welcome! If you have suggestions for new features or data cleaning operations:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Visit the project on GitHub:** [kero-emad/DataQuality](https://github.com/kero-emad/DataQuality)

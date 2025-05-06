### Project: **RoleMate AI – Intelligent Skill Matching & Upskilling Assistant**

RoleMate AI is a Streamlit-based web app that intelligently extracts and analyses skills from job descriptions and resumes using NLP and ontologies. It then recommends courses, learning paths, and related skills based on a custom ontology and external APIs.

---

### 📦 Features

* 🧠 Extracts skills from job descriptions or resumes using HuggingFace + SpaCy
* 🌐 Connects to the ESCO API for official occupation and skill mappings
* 📚 Recommends learning resources based on RDF skill ontologies
* 📈 Generates personalized learning paths
* 🔎 Auto-searches course links using SerpAPI
* 💡 Clean UI with multi-tab Streamlit interface

---

### 🗂️ Project Structure

* `streamlitApp.py`: Main Streamlit app, contains UI logic and processing
* `skill_ontology.ttl`: RDF Turtle file representing the skill ontology
* `requirements.txt`: Python dependencies
* `assets/`: Images and icons used for branding/UI
* `css/`: Optional stylesheets

---

### ⚙️ Installation

1. **Clone repo and install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download SpaCy model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Set API keys**:
   Create a `.env` file with:

   ```dotenv
   API_KEY=your_serpapi_key
   SCRAPER_API_KEY=your_scraper_api_key
   ```

---

### 🚀 Running the App

```bash
streamlit run streamlitApp.py
```

---

### 🧠 Key Technologies

* **Streamlit** – UI and state management
* **SpaCy + HuggingFace Transformers** – Skill extraction
* **RDFlib** – Ontology parsing and SPARQL querying
* **SerpAPI** – Live course search
* **ESCO API** – Official EU occupations and skills
* **PyPDF2** – Resume parsing from uploaded PDFs

---

### 📘 Usage Workflow

1. **Paste a job description** to extract required skills.
2. **Upload or paste your resume** to extract your skills.
3. **Get matched courses and personalized learning paths**.
4. **Explore related skills**, filter by difficulty or type, and start learning!

---

### ✅ Sample Inputs

* Job Description: “We’re looking for a data scientist with Python, TensorFlow, and cloud experience.”
* Resume: Upload a PDF with your work experience and skills.

---

### 🧪 Testing

To test the ontology loading or skill expansion logic in isolation, use:

```python
from rdflib import Graph
g = Graph()
g.parse("skill_ontology.ttl", format="turtle")
```



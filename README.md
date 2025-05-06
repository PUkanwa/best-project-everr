## 📁 Scripts Overview

These three scripts complement the main ontology and Streamlit app by adding, generating, and extending skill relationships and learning paths.

---

### 📜 `getjobs.py`

**Purpose**:
Adds *broader/narrower* skill relationships to the RDF ontology using `skos:broader` and `skos:narrower`.

**How It Works**:

* Loads `skill_ontology.ttl`.
* Adds semantic links between specific skill pairs (like `machine_learning` → `ai`).
* Updates the RDF graph **only if** new triples are added.
* Saves back to `skill_ontology.ttl`.

**Key Concepts**:

* Uses `rdflib` and SKOS vocab.
* Performs existence checks to prevent redundant entries.
* Logs every action clearly to terminal.

---

### 📜 `p_script.py`

**Purpose**:
Generates a **complete RDF ontology** for skills, learning resources, and personalized learning paths using course metadata (e.g., from Coursera).

**How It Works**:

* Creates classes like `Skill`, `LearningPath`, `Learning_Resource`.
* Populates the graph with skill synonyms, difficulty levels, resource types, and learning paths.
* Loads a large course dataset from a CSV file and maps each course to:

  * Taught skills
  * Ratings
  * Duration (converted to hours)
  * Provider
  * Difficulty level

**Output**:
Serializes everything to `skill_ontology-esco.ttl`.

**Key Concepts**:

* Uses `OWL`, `RDFS`, and `SKOS` extensively.
* Adds `owl:sameAs` for skill synonym handling.
* Contains 16+ curated learning paths (e.g., Python, Data Science, DevOps, SQL, UI/UX).

---

### 📜 `adding_learning_paths.py`

**Purpose**:
Adds a new **"R Programming"** learning path to an existing ontology.

**How It Works**:

* Loads `skill_ontology.ttl`.
* Adds 4 courses related to R (difficulty, duration, rating).
* Creates a `LearningPath` instance and links the courses sequentially.

**Key Concepts**:

* Automatically handles course provider creation.
* Ensures `Skill` instances are created for each course.
* Saves updated graph back to the **same file**.

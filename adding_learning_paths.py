from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
import urllib.parse

# ----------------- CONFIG -----------------
TTL_FILE = "skill_ontology.ttl"  # Existing file to update
ontology_iri = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology"
# ------------------------------------------

# Namespaces
onto = Namespace(ontology_iri + "#")
data = Namespace(ontology_iri + "/upskilling#")

# Load graph
g = Graph()
g.parse(TTL_FILE, format="turtle")
g.bind("onto", onto)
g.bind("data", data)

# Classes and properties
learning_resource_class = onto.Learning_Resource
skill_class = onto.Skill
skillTaught = onto.skillTaught
providedBy = onto.providedBy

def sanitize(text):
    return urllib.parse.quote(text.lower().replace(" ", "_").replace("&", "and"), safe="_")

# Ensure target skill
r_skill = data.r_programming
if (r_skill, RDF.type, skill_class) not in g:
    g.add((r_skill, RDF.type, skill_class))
    g.add((r_skill, RDFS.label, Literal("R Programming")))

# Real Coursera courses for R
r_courses = [
    {
        "uri": data.r_programming_course,
        "label": "R Programming",
        "provider": "Johns Hopkins University",
        "skills": ["r", "programming"],
        "difficulty": onto.BeginnerLevel,
        "rating": 4.6,
        "duration": 20.0,
    },
    {
        "uri": data.data_science_r_specialization,
        "label": "Data Science: Foundations using R Specialization",
        "provider": "Johns Hopkins University",
        "skills": ["r", "data science"],
        "difficulty": onto.IntermediateLevel,
        "rating": 4.7,
        "duration": 25.0,
    },
    {
        "uri": data.data_visualization_r,
        "label": "Data Visualization with R",
        "provider": "Google",
        "skills": ["r", "data visualization"],
        "difficulty": onto.IntermediateLevel,
        "rating": 4.6,
        "duration": 15.0,
    },
    {
        "uri": data.advanced_r_programming,
        "label": "Advanced R Programming",
        "provider": "Johns Hopkins University",
        "skills": ["r", "advanced programming"],
        "difficulty": onto.AdvancedLevel,
        "rating": 4.5,
        "duration": 20.0,
    }
]

# Add courses (if not already in graph)
for course in r_courses:
    g.add((course["uri"], RDF.type, learning_resource_class))
    g.add((course["uri"], RDFS.label, Literal(course["label"])))
    g.add((course["uri"], onto.hasDifficultyLevel, course["difficulty"]))
    g.add((course["uri"], onto.hasRating, Literal(course["rating"], datatype=XSD.decimal)))
    g.add((course["uri"], onto.estimatedDuration, Literal(course["duration"], datatype=XSD.decimal)))
    g.add((course["uri"], onto.hasResourceType, onto.Course))

    # Provider
    provider_uri = data[sanitize(course["provider"])]
    g.add((provider_uri, RDF.type, onto.CourseProvider))
    g.add((provider_uri, RDFS.label, Literal(course["provider"])))
    g.add((course["uri"], providedBy, provider_uri))

    # Skills taught
    for skill in course["skills"]:
        skill_uri = data[sanitize(skill)]
        g.add((skill_uri, RDF.type, skill_class))
        g.add((skill_uri, RDFS.label, Literal(skill.title())))
        g.add((course["uri"], skillTaught, skill_uri))

# Create and link learning path
r_path = data.r_learning_path
g.add((r_path, RDF.type, onto.LearningPath))
g.add((r_path, RDFS.label, Literal("R Programming Learning Path")))
g.add((r_path, onto.targetSkill, r_skill))
g.add((r_path, onto.hasFirstResource, r_courses[0]["uri"]))

# Link steps
for i in range(len(r_courses) - 1):
    g.add((r_courses[i]["uri"], onto.hasNextResource, r_courses[i+1]["uri"]))

# Save the updated graph **back into the same file**
g.serialize(destination=TTL_FILE, format="turtle")
print(f"✅ Updated '{TTL_FILE}' with R Programming Learning Path.")

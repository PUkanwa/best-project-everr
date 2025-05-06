from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS
import spacy
import csv
import urllib.parse
import re

# Load a model for English. Run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Define the ontology IRI
ontology_iri = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology"

# Create namespaces
onto = Namespace(ontology_iri + "#")
data = Namespace(ontology_iri + "/upskilling#")

# Add ESCO namespaces
ESCO_MODEL  = Namespace("http://data.europa.eu/esco/model#")
ESCO_SKILL = Namespace("http://data.europa.eu/esco/skill#")

# Create a new RDF Graph
g = Graph()

# Bind the namespaces to the graph
g.bind("onto", onto)
g.bind("data", data)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)
g.bind("xsd", XSD)
g.bind("skos", SKOS)  # Add SKOS vocabulary

# Add the ontology IRI to the graph
g.add((URIRef(ontology_iri), RDF.type, OWL.Ontology))
g.add(
    (
        URIRef(ontology_iri),
        RDFS.comment,
        Literal("A simple ontology for describing skills and learning resources"),
    )
)
g.add((URIRef(ontology_iri), RDFS.label, Literal("Skill Ontology (Imports ESCO)")))

# OWL:Imports for ESCO
# esco_ontology_uri = URIRef("http://data.europa.eu/esco/ontology")
esco_ontology_uri = URIRef("http://data.europa.eu/esco/skill/S")
g.add((URIRef(ontology_iri), OWL.imports, esco_ontology_uri))
print(f"Added OWL:Imports for ESCO ontology: {esco_ontology_uri}")

# Add the main classes
g.add((onto.Skill, RDF.type, OWL.Class))
g.add((onto.Learning_Resource, RDF.type, OWL.Class))
g.add((onto.CourseProvider, RDF.type, OWL.Class))
g.add((onto.CourseProvider, RDFS.label, Literal("Course Provider")))

# Define class URIs for ease of use
learning_resource_class = onto.Learning_Resource
skill_class = onto.Skill

# Define object properties
hasPrerequisite_property = onto.hasPrerequisite
skillTaught_property = onto.skillTaught
providedBy_property = onto.providedBy

g.add((hasPrerequisite_property, RDF.type, OWL.ObjectProperty))
g.add((hasPrerequisite_property, RDFS.domain, learning_resource_class))
g.add((hasPrerequisite_property, RDFS.range, skill_class))

g.add((skillTaught_property, RDF.type, OWL.ObjectProperty))
g.add((skillTaught_property, RDFS.domain, learning_resource_class))
g.add((skillTaught_property, RDFS.range, skill_class))

g.add((providedBy_property, RDF.type, OWL.ObjectProperty))
g.add((providedBy_property, RDFS.domain, learning_resource_class))
g.add((providedBy_property, RDFS.range, onto.CourseProvider))

# Define custom property for related skills
g.add((onto.relatedSkill, RDF.type, OWL.ObjectProperty))
g.add((onto.relatedSkill, RDFS.label, Literal("related skill")))
g.add((onto.relatedSkill, RDFS.domain, skill_class))
g.add((onto.relatedSkill, RDFS.range, skill_class))
g.add(
    (
        onto.relatedSkill,
        RDFS.comment,
        Literal("Connects related skills that are frequently used together"),
    )
)

# Define datatype properties for rating and target level
hasRating_property = onto.hasRating
targetLevel_property = onto.targetLevel

g.add((hasRating_property, RDF.type, OWL.DatatypeProperty))
g.add((hasRating_property, RDFS.domain, learning_resource_class))
g.add((hasRating_property, RDFS.range, XSD.decimal))

g.add((targetLevel_property, RDF.type, OWL.DatatypeProperty))
g.add((targetLevel_property, RDFS.domain, learning_resource_class))
g.add((targetLevel_property, RDFS.range, XSD.string))

# Define the HighlyRatedResource class
g.add((onto.HighlyRatedResource, RDF.type, OWL.Class))
g.add((onto.HighlyRatedResource, RDFS.subClassOf, learning_resource_class))
g.add((onto.HighlyRatedResource, RDFS.label, Literal("Highly Rated Resource")))
g.add(
    (
        onto.HighlyRatedResource,
        RDFS.comment,
        Literal("Learning resources with a rating of 4.5 or higher"),
    )
)

# Create a restriction for hasRating >= 4.5
# In OWL2, we need to create a datatype restriction and then use it in our class definition
rating_threshold = 4.5

# Create blank nodes for the restriction
restriction_node = BNode()
datatype_restriction = BNode()

# Define the datatype restriction (xsd:decimal >= 4.5)
g.add((datatype_restriction, RDF.type, RDFS.Datatype))
g.add((datatype_restriction, OWL.onDatatype, XSD.decimal))
g.add((datatype_restriction, OWL.withRestrictions, BNode("list")))

# Create the list structure for the min value restriction
g.add((BNode("list"), RDF.first, BNode("min")))
g.add((BNode("list"), RDF.rest, RDF.nil))
g.add((BNode("min"), XSD.minInclusive, Literal(rating_threshold, datatype=XSD.decimal)))

# Now create the property restriction using the datatype restriction
g.add((restriction_node, RDF.type, OWL.Restriction))
g.add((restriction_node, OWL.onProperty, hasRating_property))
g.add((restriction_node, OWL.someValuesFrom, datatype_restriction))

# Set the restriction as an equivalent class to HighlyRatedResource
g.add((onto.HighlyRatedResource, OWL.equivalentClass, restriction_node))

# ----------------------
# RESOURCE TYPE TAXONOMY
# ----------------------
# Define a class for different types of learning resources
g.add((onto.ResourceType, RDF.type, OWL.Class))
g.add((onto.ResourceType, RDFS.label, Literal("Resource Type")))
g.add(
    (
        onto.ResourceType,
        RDFS.comment,
        Literal("Categories of learning resource formats and delivery methods"),
    )
)

# Create specific resource type classes
resource_types = {
    "VideoResource": "Video-based learning content",
    "TextResource": "Text-based learning content like articles and books",
    "InteractiveResource": "Interactive exercises, quizzes, and hands-on labs",
    "Course": "Structured learning experience with multiple components",
    "Webinar": "Live or recorded web-based seminar",
    "Tutorial": "Step-by-step instructional content",
    "Documentation": "Reference materials and API documentation",
    "MOOC": "Massive Open Online Course",
    "Workshop": "Intensive, hands-on learning session",
}

# Add all resource types to the ontology
for type_name, description in resource_types.items():
    resource_type_uri = onto[type_name]
    g.add((resource_type_uri, RDF.type, OWL.Class))
    g.add((resource_type_uri, RDFS.subClassOf, onto.ResourceType))
    g.add((resource_type_uri, RDFS.label, Literal(type_name)))
    g.add((resource_type_uri, RDFS.comment, Literal(description)))

# Property to link learning resources to their type
g.add((onto.hasResourceType, RDF.type, OWL.ObjectProperty))
g.add((onto.hasResourceType, RDFS.label, Literal("has resource type")))
g.add((onto.hasResourceType, RDFS.domain, learning_resource_class))
g.add((onto.hasResourceType, RDFS.range, onto.ResourceType))

# --------------------------
# RESOURCE DURATION METADATA
# --------------------------
# Add properties for time commitment
g.add((onto.estimatedDuration, RDF.type, OWL.DatatypeProperty))
g.add((onto.estimatedDuration, RDFS.label, Literal("estimated duration")))
g.add(
    (
        onto.estimatedDuration,
        RDFS.comment,
        Literal("Estimated time to complete the resource in hours"),
    )
)
g.add((onto.estimatedDuration, RDFS.domain, learning_resource_class))
g.add((onto.estimatedDuration, RDFS.range, XSD.decimal))

# Add property for time commitment category
g.add((onto.timeCommitmentCategory, RDF.type, OWL.DatatypeProperty))
g.add((onto.timeCommitmentCategory, RDFS.label, Literal("time commitment category")))
g.add(
    (
        onto.timeCommitmentCategory,
        RDFS.comment,
        Literal("Categorized time commitment (Short: <2h, Medium: 2-10h, Long: >10h)"),
    )
)
g.add((onto.timeCommitmentCategory, RDFS.domain, learning_resource_class))
g.add((onto.timeCommitmentCategory, RDFS.range, XSD.string))

# Define an inferred class for short-duration resources
g.add((onto.ShortDurationResource, RDF.type, OWL.Class))
g.add((onto.ShortDurationResource, RDFS.subClassOf, learning_resource_class))
g.add((onto.ShortDurationResource, RDFS.label, Literal("Short Duration Resource")))

# Create a restriction: estimatedDuration <= 2.0
short_restriction = BNode()
g.add((short_restriction, RDF.type, OWL.Restriction))
g.add((short_restriction, OWL.onProperty, onto.estimatedDuration))

# Create a datatype restriction for decimal values <= 2.0
short_datatype_restriction = BNode()
g.add((short_datatype_restriction, RDF.type, RDFS.Datatype))
g.add((short_datatype_restriction, OWL.onDatatype, XSD.decimal))
g.add((short_datatype_restriction, OWL.withRestrictions, BNode("restriction_list")))

# Create the list structure for the max value restriction
g.add((BNode("restriction_list"), RDF.first, BNode("max")))
g.add((BNode("restriction_list"), RDF.rest, RDF.nil))
g.add((BNode("max"), XSD.maxInclusive, Literal(2.0, datatype=XSD.decimal)))

# Connect the restriction to the datatype restriction
g.add((short_restriction, OWL.someValuesFrom, short_datatype_restriction))

# Set the restriction as an equivalent class to ShortDurationResource
g.add((onto.ShortDurationResource, OWL.equivalentClass, short_restriction))

# ----------------------------
# RESOURCE DIFFICULTY METADATA
# ----------------------------
# Define difficulty levels class
g.add((onto.DifficultyLevel, RDF.type, OWL.Class))
g.add((onto.DifficultyLevel, RDFS.label, Literal("Difficulty Level")))

# Add the difficulty levels as individuals
difficulty_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
for level in difficulty_levels:
    diff_uri = onto[f"{level}Level"]
    g.add((diff_uri, RDF.type, OWL.NamedIndividual))
    g.add((diff_uri, RDF.type, onto.DifficultyLevel))
    g.add((diff_uri, RDFS.label, Literal(level)))

# Add property to link resources to difficulty levels
g.add((onto.hasDifficultyLevel, RDF.type, OWL.ObjectProperty))
g.add((onto.hasDifficultyLevel, RDFS.domain, learning_resource_class))
g.add((onto.hasDifficultyLevel, RDFS.range, onto.DifficultyLevel))
g.add((onto.hasDifficultyLevel, RDFS.label, Literal("has difficulty level")))

# -------------------
# LEARNING SEQUENCES
# -------------------
# Create a class for learning paths
g.add((onto.LearningPath, RDF.type, OWL.Class))
g.add((onto.LearningPath, RDFS.label, Literal("Learning Path")))
g.add(
    (
        onto.LearningPath,
        RDFS.comment,
        Literal("Structured sequence of learning resources for a specific goal"),
    )
)

# Properties for learning paths
g.add((onto.hasFirstResource, RDF.type, OWL.ObjectProperty))
g.add((onto.hasFirstResource, RDFS.domain, onto.LearningPath))
g.add((onto.hasFirstResource, RDFS.range, learning_resource_class))
g.add((onto.hasFirstResource, RDFS.label, Literal("has first resource")))

g.add((onto.hasNextResource, RDF.type, OWL.ObjectProperty))
g.add((onto.hasNextResource, RDFS.domain, learning_resource_class))
g.add((onto.hasNextResource, RDFS.range, learning_resource_class))
g.add((onto.hasNextResource, RDFS.label, Literal("has next resource")))
g.add(
    (
        onto.hasNextResource,
        RDFS.comment,
        Literal("Indicates the recommended next resource in a learning sequence"),
    )
)

# Property to indicate a learning path's target skill
g.add((onto.targetSkill, RDF.type, OWL.ObjectProperty))
g.add((onto.targetSkill, RDFS.domain, onto.LearningPath))
g.add((onto.targetSkill, RDFS.range, skill_class))
g.add((onto.targetSkill, RDFS.label, Literal("target skill")))


# Helper function to sanitize strings for URIs
def sanitize_skill(text):
    # Special case for Python - ensure consistent URI
    if text.lower() in [
        "python",
        "python programming",
        "python development",
        "python language",
    ]:
        return "python"

    safe_text = (
        text.lower().strip().replace(" ", "_").replace("&", "and").replace(":", "")
    )
    safe_text = safe_text.replace("|", "_")
    safe_text = urllib.parse.quote(safe_text, safe="_")
    return safe_text


# List of predefined skill individuals
skill_individuals = [
    # … your long list …
]

# Add the skill individuals to the graph
for skill in skill_individuals:
    skill_instance = data[sanitize_skill(skill)]
    g.add((skill_instance, RDF.type, skill_class))
    g.add((skill_instance, RDFS.label, Literal(skill)))

# Define skill synonyms and equivalence relationships
print("Adding skill synonyms and equivalence relationships...")
skill_synonyms = {
    "python": [
        "python programming",
        "python development",
        "python language",
        "python coding",
    ],
    "machine_learning": ["ml", "machine learning", "statistical learning"],
    "data_science": ["data analysis", "data analytics", "advanced analytics"],
    "javascript": ["js", "ecmascript", "javascript development"],
    "artificial_intelligence": ["ai", "artificial intelligence"],
    "java": ["java programming", "java development"],
    "web_development": ["web design", "frontend development", "website development"],
    "sql": ["database", "sql programming", "structured query language"],
    "cloud_computing": ["cloud", "cloud services", "cloud technology"],
}

# Add sameAs relationships between synonyms and SKOS labels
for main_skill, synonyms in skill_synonyms.items():
    main_uri = data[main_skill]

    # Make sure the main skill exists
    if (main_uri, RDF.type, skill_class) not in g:
        g.add((main_uri, RDF.type, skill_class))
        g.add((main_uri, RDFS.label, Literal(main_skill.replace("_", " ").title())))

    # Add SKOS preferred label
    g.add((main_uri, SKOS.prefLabel, Literal(main_skill.replace("_", " ").title())))

    # Create synonym skills and link to main skill with sameAs and SKOS
    for synonym in synonyms:
        synonym_uri = data[sanitize_skill(synonym)]

        # Create the synonym skill if it doesn't exist
        if (synonym_uri, RDF.type, skill_class) not in g:
            g.add((synonym_uri, RDF.type, skill_class))
            g.add((synonym_uri, RDFS.label, Literal(synonym)))

        # Add sameAs relationship in both directions
        g.add((synonym_uri, OWL.sameAs, main_uri))
        g.add((main_uri, OWL.sameAs, synonym_uri))

        # Add as alternative label in SKOS
        g.add((main_uri, SKOS.altLabel, Literal(synonym)))

# Create links between related skills
related_skills = [
    ("python", "data_science"),
    ("python", "machine_learning"),
    ("javascript", "web_development"),
    ("sql", "data_science"),
    ("java", "software_engineering"),
    ("machine_learning", "artificial_intelligence"),
    ("cloud_computing", "devops"),
]

for skill1, skill2 in related_skills:
    skill1_uri = data[skill1]
    skill2_uri = data[skill2]

    # Ensure both skills exist
    for skill_uri, skill_name in [(skill1_uri, skill1), (skill2_uri, skill2)]:
        if (skill_uri, RDF.type, skill_class) not in g:
            g.add((skill_uri, RDF.type, skill_class))
            g.add(
                (skill_uri, RDFS.label, Literal(skill_name.replace("_", " ").title()))
            )

    # Add bidirectional relationships
    g.add((skill1_uri, onto.relatedSkill, skill2_uri))
    g.add((skill2_uri, onto.relatedSkill, skill1_uri))

# Create a few example resources for original Python learning path
python_intro = data.introduction_to_python
python_intermediate = data.intermediate_python
python_advanced = data.advanced_python_programming

g.add((python_intro, RDF.type, learning_resource_class))
g.add((python_intro, RDFS.label, Literal("Introduction to Python")))
g.add((python_intro, onto.hasResourceType, onto.Course))
g.add((python_intro, onto.hasDifficultyLevel, onto.BeginnerLevel))

g.add((python_intermediate, RDF.type, learning_resource_class))
g.add((python_intermediate, RDFS.label, Literal("Intermediate Python Programming")))
g.add((python_intermediate, onto.hasResourceType, onto.Course))
g.add((python_intermediate, onto.hasDifficultyLevel, onto.IntermediateLevel))

g.add((python_advanced, RDF.type, learning_resource_class))
g.add((python_advanced, RDFS.label, Literal("Advanced Python Programming")))
g.add((python_advanced, onto.hasResourceType, onto.Course))
g.add((python_advanced, onto.hasDifficultyLevel, onto.AdvancedLevel))

# Create Python skill explicitly and link to courses
python_skill = data.python
python_programming_skill = data.python_programming

# Create if they don't exist
for skill_uri, label in [
    (python_skill, "Python"),
    (python_programming_skill, "Python Programming"),
]:
    if (skill_uri, RDF.type, skill_class) not in g:
        g.add((skill_uri, RDF.type, skill_class))
        g.add((skill_uri, RDFS.label, Literal(label)))

# Make them equivalent
g.add((python_skill, OWL.sameAs, python_programming_skill))
g.add((python_programming_skill, OWL.sameAs, python_skill))

# Connect both to the Python courses
python_courses = [python_intro, python_intermediate, python_advanced]
for course in python_courses:
    g.add((course, skillTaught_property, python_skill))
    g.add((course, skillTaught_property, python_programming_skill))

# Create a python learning path
python_path = data.python_learning_path
g.add((python_path, RDF.type, onto.LearningPath))
g.add((python_path, RDFS.label, Literal("Python Programming Learning Path")))
g.add(
    (python_path, onto.targetSkill, python_skill)
)  # Update to use the unified python skill

# Link resources in sequence for Python path
g.add((python_path, onto.hasFirstResource, python_intro))
g.add((python_intro, onto.hasNextResource, python_intermediate))
g.add((python_intermediate, onto.hasNextResource, python_advanced))

# Add more learning paths for different skill domains
# ----------------------------------------------------

# Define skill individuals for learning path targets if they don't exist
target_skills = [
    "data_science",
    "web_development",
    "machine_learning",
    "cloud_computing",
    "cybersecurity",
]

for skill_name in target_skills:
    skill_uri = data[skill_name]
    formatted_name = skill_name.replace("_", " ").title()
    if (skill_uri, RDF.type, skill_class) not in g:
        g.add((skill_uri, RDF.type, skill_class))
        g.add((skill_uri, RDFS.label, Literal(formatted_name)))

# 1. DATA SCIENCE LEARNING PATH
# -----------------------------
ds_path = data.data_science_learning_path
g.add((ds_path, RDF.type, onto.LearningPath))
g.add((ds_path, RDFS.label, Literal("Data Science Learning Path")))
g.add((ds_path, onto.targetSkill, data.data_science))

# Create resources for data science path
ds_intro = data.introduction_to_data_science
ds_stats = data.statistics_for_data_science
ds_viz = data.data_visualization
ds_ml_intro = data.intro_to_machine_learning_for_data_science

# Add resource metadata
for res, title, level in [
    (ds_intro, "Introduction to Data Science", onto.BeginnerLevel),
    (ds_stats, "Statistics for Data Science", onto.IntermediateLevel),
    (ds_viz, "Data Visualization Techniques", onto.IntermediateLevel),
    (ds_ml_intro, "Machine Learning for Data Scientists", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

# Link in sequence
g.add((ds_path, onto.hasFirstResource, ds_intro))
g.add((ds_intro, onto.hasNextResource, ds_stats))
g.add((ds_stats, onto.hasNextResource, ds_viz))
g.add((ds_viz, onto.hasNextResource, ds_ml_intro))

# 2. WEB DEVELOPMENT LEARNING PATH
# --------------------------------
web_path = data.web_development_learning_path
g.add((web_path, RDF.type, onto.LearningPath))
g.add((web_path, RDFS.label, Literal("Web Development Learning Path")))
g.add((web_path, onto.targetSkill, data.web_development))

# Create resources for web development path
web_html = data.html_and_css_fundamentals
web_js = data.javascript_essentials
web_react = data.react_framework
web_backend = data.backend_development

# Add resource metadata
for res, title, level in [
    (web_html, "HTML and CSS Fundamentals", onto.BeginnerLevel),
    (web_js, "JavaScript Essentials", onto.BeginnerLevel),
    (web_react, "React Framework", onto.IntermediateLevel),
    (web_backend, "Backend Development with Node.js", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

# Link in sequence
g.add((web_path, onto.hasFirstResource, web_html))
g.add((web_html, onto.hasNextResource, web_js))
g.add((web_js, onto.hasNextResource, web_react))
g.add((web_react, onto.hasNextResource, web_backend))

# 3. MACHINE LEARNING LEARNING PATH
# ---------------------------------
ml_path = data.machine_learning_learning_path
g.add((ml_path, RDF.type, onto.LearningPath))
g.add((ml_path, RDFS.label, Literal("Machine Learning Learning Path")))
g.add((ml_path, onto.targetSkill, data.machine_learning))

# Create resources for ML path
ml_intro = data.ml_foundations
ml_supervised = data.supervised_learning
ml_unsupervised = data.unsupervised_learning
ml_deep = data.deep_learning

# Add resource metadata
for res, title, level in [
    (ml_intro, "Foundations of Machine Learning", onto.BeginnerLevel),
    (ml_supervised, "Supervised Learning Algorithms", onto.IntermediateLevel),
    (ml_unsupervised, "Unsupervised Learning", onto.IntermediateLevel),
    (ml_deep, "Deep Learning with Neural Networks", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))
    # Add duration (example values)
    g.add(
        (
            res,
            onto.estimatedDuration,
            Literal(
                float(level == onto.BeginnerLevel and 5 or 8), datatype=XSD.decimal
            ),
        )
    )

# Link in sequence
g.add((ml_path, onto.hasFirstResource, ml_intro))
g.add((ml_intro, onto.hasNextResource, ml_supervised))
g.add((ml_supervised, onto.hasNextResource, ml_unsupervised))
g.add((ml_unsupervised, onto.hasNextResource, ml_deep))

# 4. CLOUD COMPUTING LEARNING PATH
# --------------------------------
cloud_path = data.cloud_computing_learning_path
g.add((cloud_path, RDF.type, onto.LearningPath))
g.add((cloud_path, RDFS.label, Literal("Cloud Computing Learning Path")))
g.add((cloud_path, onto.targetSkill, data.cloud_computing))

# Create resources for cloud path
cloud_intro = data.cloud_fundamentals
cloud_aws = data.aws_essentials
cloud_azure = data.azure_services
cloud_devops = data.cloud_devops

# Add resource metadata
for res, title, level in [
    (cloud_intro, "Cloud Computing Fundamentals", onto.BeginnerLevel),
    (cloud_aws, "AWS Essentials", onto.IntermediateLevel),
    (cloud_azure, "Microsoft Azure Services", onto.IntermediateLevel),
    (cloud_devops, "DevOps for Cloud Environments", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

# Link in sequence
g.add((cloud_path, onto.hasFirstResource, cloud_intro))
g.add((cloud_intro, onto.hasNextResource, cloud_aws))
g.add((cloud_aws, onto.hasNextResource, cloud_azure))
g.add((cloud_azure, onto.hasNextResource, cloud_devops))

# 5. CYBERSECURITY LEARNING PATH
# ------------------------------
security_path = data.cybersecurity_learning_path
g.add((security_path, RDF.type, onto.LearningPath))
g.add((security_path, RDFS.label, Literal("Cybersecurity Learning Path")))
g.add((security_path, onto.targetSkill, data.cybersecurity))

# Create resources for cybersecurity path
sec_intro = data.security_fundamentals
sec_network = data.network_security
sec_cryptography = data.cryptography_essentials
sec_pentesting = data.penetration_testing

# Add resource metadata and connect skills with the resources
for res, title, level, skills in [
    (
        sec_intro,
        "Cybersecurity Fundamentals",
        onto.BeginnerLevel,
        ["cybersecurity", "information_security"],
    ),
    (
        sec_network,
        "Network Security",
        onto.IntermediateLevel,
        ["network_security", "firewalls"],
    ),
    (
        sec_cryptography,
        "Cryptography Essentials",
        onto.IntermediateLevel,
        ["cryptography", "encryption"],
    ),
    (
        sec_pentesting,
        "Penetration Testing",
        onto.AdvancedLevel,
        ["penetration_testing", "ethical_hacking"],
    ),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

    # Link the skills taught by this resource
    for skill in skills:
        skill_uri = data[skill]
        # Create the skill if it doesn't exist
        if (skill_uri, RDF.type, skill_class) not in g:
            g.add((skill_uri, RDF.type, skill_class))
            g.add((skill_uri, RDFS.label, Literal(skill.replace("_", " ").title())))
        # Link skill to resource
        g.add((res, skillTaught_property, skill_uri))

# Link in sequence
g.add((security_path, onto.hasFirstResource, sec_intro))
g.add((sec_intro, onto.hasNextResource, sec_network))
g.add((sec_network, onto.hasNextResource, sec_cryptography))
g.add((sec_cryptography, onto.hasNextResource, sec_pentesting))

# ----------------------------------------------------
# 6. DEVOPS LEARNING PATH
# ----------------------------------------------------
devops_path = data.devops_learning_path
g.add((devops_path, RDF.type, onto.LearningPath))
g.add((devops_path, RDFS.label, Literal("DevOps Learning Path")))
g.add((devops_path, onto.targetSkill, data.devops))

devops_intro = data.introduction_to_devops
devops_ci_cd = data.ci_cd_practices
devops_containers = data.containers_and_docker
devops_k8s = data.kubernetes_fundamentals
devops_iac = data.infrastructure_as_code

for res, title, level in [
    (devops_intro, "Introduction to DevOps", onto.BeginnerLevel),
    (devops_ci_cd, "CI/CD Practices", onto.IntermediateLevel),
    (devops_containers, "Containers and Docker", onto.IntermediateLevel),
    (devops_k8s, "Kubernetes Fundamentals", onto.AdvancedLevel),
    (devops_iac, "Infrastructure as Code with Terraform", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((devops_path, onto.hasFirstResource, devops_intro))
g.add((devops_intro, onto.hasNextResource, devops_ci_cd))
g.add((devops_ci_cd, onto.hasNextResource, devops_containers))
g.add((devops_containers, onto.hasNextResource, devops_k8s))
g.add((devops_k8s, onto.hasNextResource, devops_iac))


# ----------------------------------------------------
# 7. BIG DATA LEARNING PATH
# ----------------------------------------------------
bigdata_path = data.big_data_learning_path
g.add((bigdata_path, RDF.type, onto.LearningPath))
g.add((bigdata_path, RDFS.label, Literal("Big Data Learning Path")))
g.add((bigdata_path, onto.targetSkill, data.big_data))

bd_hadoop = data.hadoop_essentials
bd_spark = data.spark_fundamentals
bd_warehouse = data.data_warehousing
bd_stream = data.real_time_data_streaming
bd_ml_big = data.machine_learning_at_scale

for res, title, level in [
    (bd_hadoop, "Hadoop Essentials", onto.BeginnerLevel),
    (bd_spark, "Apache Spark Fundamentals", onto.IntermediateLevel),
    (bd_warehouse, "Data Warehousing Concepts", onto.IntermediateLevel),
    (bd_stream, "Real-Time Data Streaming with Kafka", onto.AdvancedLevel),
    (bd_ml_big, "Machine Learning at Scale", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((bigdata_path, onto.hasFirstResource, bd_hadoop))
g.add((bd_hadoop, onto.hasNextResource, bd_spark))
g.add((bd_spark, onto.hasNextResource, bd_warehouse))
g.add((bd_warehouse, onto.hasNextResource, bd_stream))
g.add((bd_stream, onto.hasNextResource, bd_ml_big))


# ----------------------------------------------------
# 8. UI/UX DESIGN LEARNING PATH
# ----------------------------------------------------
ux_path = data.ui_ux_design_learning_path
g.add((ux_path, RDF.type, onto.LearningPath))
g.add((ux_path, RDFS.label, Literal("UI/UX Design Learning Path")))
g.add((ux_path, onto.targetSkill, data.ui_ux_design))

ux_research = data.user_research_and_usability
ux_wireframe = data.wireframing_and_prototyping
ux_visual = data.visual_design_principles
ux_interaction = data.interaction_design
ux_testing = data.usability_testing_methods

for res, title, level in [
    (ux_research, "User Research & Usability", onto.BeginnerLevel),
    (ux_wireframe, "Wireframing & Prototyping", onto.IntermediateLevel),
    (ux_visual, "Visual Design Principles", onto.IntermediateLevel),
    (ux_interaction, "Interaction Design Essentials", onto.AdvancedLevel),
    (ux_testing, "Usability Testing Methods", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((ux_path, onto.hasFirstResource, ux_research))
g.add((ux_research, onto.hasNextResource, ux_wireframe))
g.add((ux_wireframe, onto.hasNextResource, ux_visual))
g.add((ux_visual, onto.hasNextResource, ux_interaction))
g.add((ux_interaction, onto.hasNextResource, ux_testing))


# ----------------------------------------------------
# 9. PROJECT MANAGEMENT LEARNING PATH
# ----------------------------------------------------
pm_path = data.project_management_learning_path
g.add((pm_path, RDF.type, onto.LearningPath))
g.add((pm_path, RDFS.label, Literal("Project Management Learning Path")))
g.add((pm_path, onto.targetSkill, data.project_management))

pm_intro = data.project_management_fundamentals
pm_agile = data.agile_methodologies
pm_scrum = data.scrum_master_practices
pm_risk = data.project_risk_management
pm_pmp = data.pmp_certification_prep

for res, title, level in [
    (pm_intro, "Project Management Fundamentals", onto.BeginnerLevel),
    (pm_agile, "Agile Methodologies", onto.IntermediateLevel),
    (pm_scrum, "Scrum Master Practices", onto.IntermediateLevel),
    (pm_risk, "Project Risk Management", onto.AdvancedLevel),
    (pm_pmp, "PMP Certification Prep", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((pm_path, onto.hasFirstResource, pm_intro))
g.add((pm_intro, onto.hasNextResource, pm_agile))
g.add((pm_agile, onto.hasNextResource, pm_scrum))
g.add((pm_scrum, onto.hasNextResource, pm_risk))
g.add((pm_risk, onto.hasNextResource, pm_pmp))


# ----------------------------------------------------
# 10. QUALITY ASSURANCE LEARNING PATH
# ----------------------------------------------------
qa_path = data.qa_learning_path
g.add((qa_path, RDF.type, onto.LearningPath))
g.add((qa_path, RDFS.label, Literal("Quality Assurance Learning Path")))
g.add((qa_path, onto.targetSkill, data.quality_assurance))

qa_fund = data.qa_fundamentals
qa_test_auto = data.test_automation_with_selenium
qa_api = data.api_testing_methods
qa_perf = data.performance_testing_techniques
qa_sec = data.security_testing_basics

for res, title, level in [
    (qa_fund, "QA Fundamentals", onto.BeginnerLevel),
    (qa_test_auto, "Test Automation with Selenium", onto.IntermediateLevel),
    (qa_api, "API Testing Methods", onto.IntermediateLevel),
    (qa_perf, "Performance Testing Techniques", onto.AdvancedLevel),
    (qa_sec, "Security Testing Basics", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((qa_path, onto.hasFirstResource, qa_fund))
g.add((qa_fund, onto.hasNextResource, qa_test_auto))
g.add((qa_test_auto, onto.hasNextResource, qa_api))
g.add((qa_api, onto.hasNextResource, qa_perf))
g.add((qa_perf, onto.hasNextResource, qa_sec))


# ----------------------------------------------------
# 11. JAVA LEARNING PATH
# ----------------------------------------------------
java_path = data.java_learning_path
g.add((java_path, RDF.type, onto.LearningPath))
g.add((java_path, RDFS.label, Literal("Java Learning Path")))
g.add((java_path, onto.targetSkill, data.java))

java_basics = data.java_basics
java_oop = data.object_oriented_programming_in_java
java_collections = data.java_collections_framework
java_concurrency = data.java_concurrency
java_advanced = data.advanced_java_features

for res, title, level in [
    (java_basics, "Java Basics: Syntax & Variables", onto.BeginnerLevel),
    (java_oop, "Object-Oriented Programming in Java", onto.IntermediateLevel),
    (java_collections, "Java Collections Framework", onto.IntermediateLevel),
    (java_concurrency, "Java Concurrency & Multithreading", onto.AdvancedLevel),
    (java_advanced, "Advanced Java Features & JVM Internals", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((java_path, onto.hasFirstResource, java_basics))
g.add((java_basics, onto.hasNextResource, java_oop))
g.add((java_oop, onto.hasNextResource, java_collections))
g.add((java_collections, onto.hasNextResource, java_concurrency))
g.add((java_concurrency, onto.hasNextResource, java_advanced))


# ----------------------------------------------------
# 12. JAVASCRIPT LEARNING PATH
# ----------------------------------------------------
js_path = data.javascript_learning_path
g.add((js_path, RDF.type, onto.LearningPath))
g.add((js_path, RDFS.label, Literal("JavaScript Learning Path")))
g.add((js_path, onto.targetSkill, data.javascript))

js_fundamentals = data.javascript_fundamentals
js_dom = data.dom_manipulation_and_events
js_async = data.asynchronous_javascript_promises_async
js_frameworks = data.frontend_frameworks_react_vue
js_advanced = data.advanced_javascript_patterns

for res, title, level in [
    (js_fundamentals, "JavaScript Fundamentals & ES6+", onto.BeginnerLevel),
    (js_dom, "DOM Manipulation & Event Handling", onto.IntermediateLevel),
    (js_async, "Asynchronous JS: Callbacks, Promises & Async", onto.IntermediateLevel),
    (js_frameworks, "Frontend Frameworks: React & Vue", onto.AdvancedLevel),
    (js_advanced, "Advanced JS Patterns & Performance", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((js_path, onto.hasFirstResource, js_fundamentals))
g.add((js_fundamentals, onto.hasNextResource, js_dom))
g.add((js_dom, onto.hasNextResource, js_async))
g.add((js_async, onto.hasNextResource, js_frameworks))
g.add((js_frameworks, onto.hasNextResource, js_advanced))


# ----------------------------------------------------
# 13. C++ LEARNING PATH
# ----------------------------------------------------
cpp_path = data.cpp_learning_path
g.add((cpp_path, RDF.type, onto.LearningPath))
g.add((cpp_path, RDFS.label, Literal("C++ Learning Path")))
g.add((cpp_path, onto.targetSkill, data.cpp))

cpp_basics = data.cpp_basics
cpp_oop = data.cpp_object_oriented_programming
cpp_stl = data.stl_and_generic_programming
cpp_memory = data.dynamic_memory_management_cpp
cpp_modern = data.modern_cpp_features

for res, title, level in [
    (cpp_basics, "C++ Basics: Syntax & Variables", onto.BeginnerLevel),
    (cpp_oop, "OOP in C++: Classes & Inheritance", onto.IntermediateLevel),
    (cpp_stl, "STL & Generic Programming", onto.IntermediateLevel),
    (cpp_memory, "Dynamic Memory & Resource Management", onto.AdvancedLevel),
    (cpp_modern, "Modern C++ (11/14/17/20) Features", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((cpp_path, onto.hasFirstResource, cpp_basics))
g.add((cpp_basics, onto.hasNextResource, cpp_oop))
g.add((cpp_oop, onto.hasNextResource, cpp_stl))
g.add((cpp_stl, onto.hasNextResource, cpp_memory))
g.add((cpp_memory, onto.hasNextResource, cpp_modern))


# ----------------------------------------------------
# 14. SQL LEARNING PATH
# ----------------------------------------------------
sql_path = data.sql_learning_path
g.add((sql_path, RDF.type, onto.LearningPath))
g.add((sql_path, RDFS.label, Literal("SQL Learning Path")))
g.add((sql_path, onto.targetSkill, data.sql))

sql_basics = data.sql_basics
sql_joins = data.advanced_sql_joins
sql_indexes = data.indexing_and_performance
sql_subqueries = data.subqueries_and_ctes
sql_advanced = data.advanced_database_design

for res, title, level in [
    (sql_basics, "SQL Basics: SELECT, INSERT, UPDATE", onto.BeginnerLevel),
    (sql_joins, "Advanced Joins & Set Operations", onto.IntermediateLevel),
    (sql_indexes, "Indexing & Query Performance Tuning", onto.IntermediateLevel),
    (sql_subqueries, "Subqueries, CTEs & Window Functions", onto.AdvancedLevel),
    (sql_advanced, "Advanced Database Design & Normalization", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((sql_path, onto.hasFirstResource, sql_basics))
g.add((sql_basics, onto.hasNextResource, sql_joins))
g.add((sql_joins, onto.hasNextResource, sql_indexes))
g.add((sql_indexes, onto.hasNextResource, sql_subqueries))
g.add((sql_subqueries, onto.hasNextResource, sql_advanced))


# ----------------------------------------------------
# 15. DATA ANALYSIS LEARNING PATH
# ----------------------------------------------------
da_path = data.data_analysis_learning_path
g.add((da_path, RDF.type, onto.LearningPath))
g.add((da_path, RDFS.label, Literal("Data Analysis Learning Path")))
g.add((da_path, onto.targetSkill, data.data_analysis))

da_collect = data.data_collection_and_cleaning
da_eda = data.exploratory_data_analysis
da_stats = data.statistical_analysis_for_data
da_htest = data.hypothesis_testing_and_inference
da_tools = data.data_visualization_tools

for res, title, level in [
    (da_collect, "Data Collection & Cleaning Techniques", onto.BeginnerLevel),
    (da_eda, "Exploratory Data Analysis (EDA)", onto.IntermediateLevel),
    (da_stats, "Statistical Analysis for Data Insights", onto.IntermediateLevel),
    (da_htest, "Hypothesis Testing & Statistical Inference", onto.AdvancedLevel),
    (da_tools, "Data Visualization with Python & R", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((da_path, onto.hasFirstResource, da_collect))
g.add((da_collect, onto.hasNextResource, da_eda))
g.add((da_eda, onto.hasNextResource, da_stats))
g.add((da_stats, onto.hasNextResource, da_htest))
g.add((da_htest, onto.hasNextResource, da_tools))


# ----------------------------------------------------
# 16. WEB FRAMEWORKS LEARNING PATH (React & Angular)
# ----------------------------------------------------
wf_path = data.web_frameworks_learning_path
g.add((wf_path, RDF.type, onto.LearningPath))
g.add((wf_path, RDFS.label, Literal("Web Frameworks Learning Path")))
g.add((wf_path, onto.targetSkill, data.web_development))

wf_react = data.react_essentials
wf_angular = data.angular_basics
wf_state = data.state_management_redux_ngrx
wf_ssr = data.server_side_rendering_next_nest
wf_testing = data.frontend_testing_and_ci_cd

for res, title, level in [
    (wf_react, "React Essentials: JSX & Components", onto.BeginnerLevel),
    (wf_angular, "Angular Basics: Modules & Templates", onto.BeginnerLevel),
    (wf_state, "State Management with Redux & NgRx", onto.IntermediateLevel),
    (wf_ssr, "Server-Side Rendering with Next.js & NestJS", onto.AdvancedLevel),
    (wf_testing, "Frontend Testing & CI/CD Pipelines", onto.AdvancedLevel),
]:
    g.add((res, RDF.type, learning_resource_class))
    g.add((res, RDFS.label, Literal(title)))
    g.add((res, onto.hasResourceType, onto.Course))
    g.add((res, onto.hasDifficultyLevel, level))

g.add((wf_path, onto.hasFirstResource, wf_react))
g.add((wf_react, onto.hasNextResource, wf_angular))
g.add((wf_angular, onto.hasNextResource, wf_state))
g.add((wf_state, onto.hasNextResource, wf_ssr))
g.add((wf_ssr, onto.hasNextResource, wf_testing))


# Dictionary to store CourseProvider instances to avoid duplicates
providers = {}

# Dictionary to track skills we've already created to avoid duplicates
created_skills = {}

# Read the CSV
with open(
    r"C:\Users\ukanw\Downloads\coursera_course_dataset_v2_no_null.csv",
    "r",
    encoding="utf-8",
) as f:
    reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)
    for row in reader:
        title = row.get("Title", "").strip()
        provider_name = row.get("Organization", "").strip()
        skills_str = row.get("Skills", "").strip()
        rating_str = row.get("Ratings", "").strip()
        level_str = row.get(
            "Metadata", ""
        ).strip()  # e.g. "Beginner · Professional Certificate · 3 - 6 Months"

        # Create a learning resource instance
        resource_instance = data[sanitize_skill(title)]
        g.add((resource_instance, RDF.type, learning_resource_class))
        g.add((resource_instance, RDFS.label, Literal(title)))

        # Link to provider
        if provider_name not in providers:
            provider_uri = data[sanitize_skill(provider_name)]
            g.add((provider_uri, RDF.type, onto.CourseProvider))
            g.add((provider_uri, RDFS.label, Literal(provider_name)))
            providers[provider_name] = provider_uri
        else:
            provider_uri = providers[provider_name]
        g.add((resource_instance, providedBy_property, provider_uri))

        # Link to skills taught
        for skill_name in [s.strip() for s in skills_str.split(",") if s.strip()]:
            skill_uri = data[sanitize_skill(skill_name)]

            # Create the skill as an individual of the Skill class (if not already created)
            if skill_name not in created_skills:
                g.add((skill_uri, RDF.type, skill_class))
                g.add((skill_uri, RDFS.label, Literal(skill_name)))
                created_skills[skill_name] = skill_uri

            # Link the skill to the resource
            g.add((resource_instance, skillTaught_property, skill_uri))

            # Special case for Python courses - ensure they're linked to both Python skill URIs
            if "python" in skill_name.lower():
                g.add((resource_instance, skillTaught_property, data.python))
                g.add(
                    (resource_instance, skillTaught_property, data.python_programming)
                )

        # Add rating if available
        if rating_str:
            try:
                rating_val = float(rating_str)
                g.add(
                    (
                        resource_instance,
                        hasRating_property,
                        Literal(rating_val, datatype=XSD.decimal),
                    )
                )

                # Manually classify highly rated resources for systems that don't perform inference
                if rating_val >= 4.5:
                    g.add((resource_instance, RDF.type, onto.HighlyRatedResource))

            except ValueError:
                pass

        # Process metadata for difficulty level and duration
        if level_str:
            # Extract difficulty level
            level = level_str.split("·")[0].strip()
            if level:
                g.add(
                    (
                        resource_instance,
                        targetLevel_property,
                        Literal(level, datatype=XSD.string),
                    )
                )

                # Map to difficulty level individuals
                if "beginner" in level.lower():
                    g.add(
                        (resource_instance, onto.hasDifficultyLevel, onto.BeginnerLevel)
                    )
                elif "intermediate" in level.lower():
                    g.add(
                        (
                            resource_instance,
                            onto.hasDifficultyLevel,
                            onto.IntermediateLevel,
                        )
                    )
                elif "advanced" in level.lower():
                    g.add(
                        (resource_instance, onto.hasDifficultyLevel, onto.AdvancedLevel)
                    )
                elif "expert" in level.lower():
                    g.add(
                        (resource_instance, onto.hasDifficultyLevel, onto.ExpertLevel)
                    )

            # Try to determine resource type from metadata or title
            if "course" in title.lower():
                g.add((resource_instance, onto.hasResourceType, onto.Course))
            elif "specialization" in title.lower():
                g.add((resource_instance, onto.hasResourceType, onto.Course))
            elif "workshop" in title.lower():
                g.add((resource_instance, onto.hasResourceType, onto.Workshop))
            elif "tutorial" in title.lower():
                g.add((resource_instance, onto.hasResourceType, onto.Tutorial))
            else:
                # Default type for Coursera data
                g.add((resource_instance, onto.hasResourceType, onto.MOOC))

            # Extract duration information if available
            # Example pattern: "3 - 6 Months"
            duration_match = re.search(
                r"(\d+)\s*-\s*(\d+)\s*(Month|Week|Hour|Day)", level_str, re.IGNORECASE
            )
            if duration_match:
                avg_duration = (
                    float(duration_match.group(1)) + float(duration_match.group(2))
                ) / 2
                unit = duration_match.group(3).lower()

                # Convert to hours (approximate)
                if unit.startswith("month"):
                    hours = (
                        avg_duration * 30
                    )  # ~30 hours per month (assuming part-time study)
                elif unit.startswith("week"):
                    hours = avg_duration * 7  # ~7 hours per week
                elif unit.startswith("day"):
                    hours = avg_duration * 1  # ~1 hour per day
                else:  # already in hours
                    hours = avg_duration

                g.add(
                    (
                        resource_instance,
                        onto.estimatedDuration,
                        Literal(hours, datatype=XSD.decimal),
                    )
                )

                # Add time commitment category
                if hours <= 2:
                    g.add(
                        (
                            resource_instance,
                            onto.timeCommitmentCategory,
                            Literal("Short", datatype=XSD.string),
                        )
                    )
                elif hours <= 10:
                    g.add(
                        (
                            resource_instance,
                            onto.timeCommitmentCategory,
                            Literal("Medium", datatype=XSD.string),
                        )
                    )
                else:
                    g.add(
                        (
                            resource_instance,
                            onto.timeCommitmentCategory,
                            Literal("Long", datatype=XSD.string),
                        )
                    )

# Add hard-coded Python courses for fallback
python_courses_hardcoded = [
    {
        "title": "Python for Everybody Specialization",
        "provider": "University of Michigan",
        "skills": ["python", "programming"],
        "rating": 4.8,
        "difficulty": "Beginner",
        "duration": 20.0,
    },
    {
        "title": "Python 3 Programming Specialization",
        "provider": "University of Michigan",
        "skills": ["python", "programming"],
        "rating": 4.7,
        "difficulty": "Beginner",
        "duration": 25.0,
    },
    {
        "title": "Applied Data Science with Python",
        "provider": "University of Michigan",
        "skills": ["python", "data science"],
        "rating": 4.6,
        "difficulty": "Intermediate",
        "duration": 30.0,
    },
]

for course_info in python_courses_hardcoded:
    course_uri = data[sanitize_skill(course_info["title"])]

    # Skip if already exists
    if (course_uri, RDF.type, learning_resource_class) in g:
        continue

    # Add the course
    g.add((course_uri, RDF.type, learning_resource_class))
    g.add((course_uri, RDFS.label, Literal(course_info["title"])))

    # Add provider
    provider_name = course_info["provider"]
    if provider_name not in providers:
        provider_uri = data[sanitize_skill(provider_name)]
        g.add((provider_uri, RDF.type, onto.CourseProvider))
        g.add((provider_uri, RDFS.label, Literal(provider_name)))
        providers[provider_name] = provider_uri
    else:
        provider_uri = providers[provider_name]
    g.add((course_uri, providedBy_property, provider_uri))

    # Link skills
    for skill_name in course_info["skills"]:
        skill_uri = data[sanitize_skill(skill_name)]
        if (skill_uri, RDF.type, skill_class) not in g:
            g.add((skill_uri, RDF.type, skill_class))
            g.add((skill_uri, RDFS.label, Literal(skill_name.title())))
        g.add((course_uri, skillTaught_property, skill_uri))

    # Add rating
    g.add(
        (
            course_uri,
            hasRating_property,
            Literal(course_info["rating"], datatype=XSD.decimal),
        )
    )

    # Add difficulty
    if "beginner" in course_info["difficulty"].lower():
        g.add((course_uri, onto.hasDifficultyLevel, onto.BeginnerLevel))
    elif "intermediate" in course_info["difficulty"].lower():
        g.add((course_uri, onto.hasDifficultyLevel, onto.IntermediateLevel))
    elif "advanced" in course_info["difficulty"].lower():
        g.add((course_uri, onto.hasDifficultyLevel, onto.AdvancedLevel))

    # Add duration
    g.add(
        (
            course_uri,
            onto.estimatedDuration,
            Literal(course_info["duration"], datatype=XSD.decimal),
        )
    )

    # Add resource type
    g.add((course_uri, onto.hasResourceType, onto.Course))

    # Make sure it's linked to both Python skill URIs
    g.add((course_uri, skillTaught_property, data.python))
    g.add((course_uri, skillTaught_property, data.python_programming))

# Serialize the graph to a file
output_file = "skill_ontology-esco.ttl"
g.serialize(destination=output_file, format="turtle")
print(f"Skill ontology created and saved to {output_file}")

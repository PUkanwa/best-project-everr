from rdflib import Graph, Namespace
import os

ontology_file_path = "skill_ontology.ttl"
ONTO = Namespace("http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#")

DATA = Namespace(
    "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology/upskilling#"
)
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")


relationships_to_add = [
    ("machine_learning", "ai"),
    ("data_visualization", "data_science"),
    ("deep_learning", "machine_learning"),
    ("sql", "database_administration"),
]

print(f"Loading ontology from: {ontology_file_path}")

if not os.path.exists(ontology_file_path):
    print(f"Error: Ontology file not found at '{ontology_file_path}'")
    exit()

g = Graph()
try:
    g.parse(ontology_file_path, format="turtle")
    print(f"Ontology loaded successfully. Graph has {len(g)} triples.")
except Exception as e:
    print(f"Error parsing ontology file: {e}")
    exit()

g.bind("onto", ONTO)
g.bind("data", DATA)
g.bind("skos", SKOS)
g.bind("rdfs", RDFS)
g.bind("rdf", RDF)

changes_made = False

for narrower_name, broader_name in relationships_to_add:
    print(f"\nProcessing relationship: '{narrower_name}' skos:broader '{broader_name}'")

    narrower_uri = DATA[narrower_name]
    broader_uri = DATA[broader_name]

    if (narrower_uri, RDF.type, ONTO.Skill) not in g:
        print(
            f"  Warning: Narrower skill <{narrower_uri}> not found or not typed as onto:Skill. Skipping relationship."
        )
        continue
    if (broader_uri, RDF.type, ONTO.Skill) not in g:
        print(
            f"  Warning: Broader skill <{broader_uri}> not found or not typed as onto:Skill. Skipping relationship."
        )
        continue

    broader_triple = (narrower_uri, SKOS.broader, broader_uri)

    narrower_triple = (broader_uri, SKOS.narrower, narrower_uri)

    if broader_triple not in g:
        g.add(broader_triple)
        print(f"  Added: <{narrower_uri.n3()}> skos:broader <{broader_uri.n3()}> .")
        changes_made = True
    else:
        print(f"  Exists: <{narrower_uri.n3()}> skos:broader <{broader_uri.n3()}> .")

    if narrower_triple not in g:
        g.add(narrower_triple)
        print(f"  Added: <{broader_uri.n3()}> skos:narrower <{narrower_uri.n3()}> .")
        changes_made = True
    else:
        print(f"  Exists: <{broader_uri.n3()}> skos:narrower <{narrower_uri.n3()}> .")


if changes_made:
    try:
        print(f"\nSaving updated ontology back to: {ontology_file_path}")

        g.serialize(destination=ontology_file_path, format="turtle", encoding="utf-8")
        print("Ontology saved successfully.")
    except Exception as e:
        print(f"Error saving updated ontology file: {e}")
else:
    print("\nNo changes were made to the ontology.")

print("Script finished.")

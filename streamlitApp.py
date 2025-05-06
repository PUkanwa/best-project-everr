import streamlit as st
from rdflib import Graph, Namespace, RDF, RDFS, SKOS
import spacy
import time
from transformers import pipeline
import urllib.parse
import requests
import concurrent.futures
from serpapi import GoogleSearch
import traceback
import os
import base64
import PyPDF2
from dotenv import load_dotenv
from string import capwords

load_dotenv()
SERPAPI_API_KEY = os.getenv("API_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")


def load_css(css_file="css/streamlitApp.css"):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{css_file}' not found. Using default styling.")


if "url_validation_cache" not in st.session_state:
    st.session_state["url_validation_cache"] = {}
if "link_cache" not in st.session_state:
    st.session_state["link_cache"] = {}
if "cache_last_cleaned" not in st.session_state:
    st.session_state["cache_last_cleaned"] = time.time()


def manage_cache_automatically():
    current_time = time.time()
    if current_time - st.session_state["cache_last_cleaned"] > 86400:
        if len(st.session_state["url_validation_cache"]) > 2000:
            st.session_state["url_validation_cache"] = dict(
                list(st.session_state["url_validation_cache"].items())[-1000:]
            )

        if len(st.session_state["link_cache"]) > 2000:
            st.session_state["link_cache"] = dict(
                list(st.session_state["link_cache"].items())[-1000:]
            )
        st.session_state["cache_last_cleaned"] = current_time


manage_cache_automatically()

TOP_PROVIDERS = [
    "coursera",
    "udemy",
    "edx",
    "pluralsight",
    "linkedin learning",
    "udacity",
    "google",
    "microsoft",
    "aws",
    "ibm",
    "meta",
    "deeplearning.ai",
]


def sanitise_skill(text):
    safe_text = (
        text.lower()
        .strip()
        .replace(" ", "_")
        .replace("&", "and")
        .replace(":", "")
        .replace("|", "_")
    )
    return urllib.parse.quote(safe_text, safe="_")


def get_difficulty_badge(difficulty):
    """Returns HTML for a difficulty badge with appropriate color"""
    if not difficulty:
        return ""

    difficulty = difficulty.lower()
    if "beginner" in difficulty:
        color = "#28a745"
    elif "intermediate" in difficulty:
        color = "#ffc107"
    elif "advanced" in difficulty:
        color = "#fd7e14"  # Orange
    elif "expert" in difficulty:
        color = "#dc3545"  # Red
    else:
        color = "#6c757d"  # Gray

    return f"""
    <span style="display:inline-block; padding:3px 8px; border-radius:12px; 
                background-color:{color}; color:white; font-size:0.8em; 
                font-weight:bold;">
        {difficulty.title()}
    </span>
    """


def search_course_link_serpapi(course, provider, api_key):
    params = {
        "api_key": api_key,
        "engine": "google",
        "q": f"{course} {provider} online course",
        "num": 1,
    }

    # Check if API key is available
    if not api_key:
        print("Missing SerpAPI key - cannot search for course links")
        return None

    try:
        # Print what we're searching (for debugging)
        print(f"Searching for: {course} by {provider}")

        search = GoogleSearch(params)
        results = search.get_dict()

        # Debug the response
        if "organic_results" in results and len(results["organic_results"]) > 0:
            link = results["organic_results"][0]["link"]
            print(f"Found link: {link}")
            return link
        else:
            print(f"No organic results found for {course} by {provider}")
            # Check if there's an error message in the response
            if "error" in results:
                print(f"SerpAPI error: {results['error']}")
            return None

    except Exception as e:
        print(f"\n!!! SerpAPI search error for '{course}' by '{provider}' !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        return None


def search_esco_occupations_api(job_title):
    """
    Two-step ESCO API search: first find the occupation, then get related occupations.

    Args:
        job_title (str): The job title to search for

    Returns:
        dict: Information about the occupation and similar roles
    """
    # Step 1: Search for the occupation by title
    search_url = "https://ec.europa.eu/esco/api/search"
    search_params = {
        "language": "en",
        "type": "occupation",
        "text": job_title,
        "selectedVersion": "v1.2.0",
    }

    try:
        response = requests.get(search_url, params=search_params, timeout=10)
        print(f"Search response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if (
                "_embedded" in data
                and "results" in data["_embedded"]
                and data["_embedded"]["results"]
            ):
                first_result = data["_embedded"]["results"][0]
                occupation_uri = first_result.get("uri")

                if not occupation_uri:
                    print("No URI found in first result")
                    return get_fallback_occupations(job_title)

                print(f"Found occupation: {first_result.get('title')}")
                print(f"Occupation URI: {occupation_uri}")

                detail_url = "https://ec.europa.eu/esco/api/resource/occupation"
                detail_params = {"language": "en", "uri": occupation_uri}

                detail_response = requests.get(
                    detail_url, params=detail_params, timeout=10
                )
                print(f"Detail response status: {detail_response.status_code}")

                if detail_response.status_code == 200:
                    occupation_data = detail_response.json()

                    result = {
                        "title": occupation_data.get(
                            "title", first_result.get("title", "")
                        ),
                        "description": occupation_data.get("description", {})
                        .get("en", {})
                        .get("literal", ""),
                        "uri": occupation_uri,
                        "similarOccupations": [],
                    }

                    if "broaderOccupation" in occupation_data:
                        for broader in occupation_data["broaderOccupation"]:
                            result["similarOccupations"].append(
                                {
                                    "title": broader.get("title", ""),
                                    "relation": "Broader role",
                                }
                            )

                    if "narrowerOccupation" in occupation_data:
                        for narrower in occupation_data["narrowerOccupation"]:
                            result["similarOccupations"].append(
                                {
                                    "title": narrower.get("title", ""),
                                    "relation": "More specialised role",
                                }
                            )

                    for alt_result in data["_embedded"]["results"][1:6]:
                        if alt_result.get("title") != result["title"]:
                            result["similarOccupations"].append(
                                {
                                    "title": alt_result.get("title", ""),
                                    "relation": "Alternative role",
                                }
                            )

                    if result["similarOccupations"]:
                        return result
                    else:
                        for alt_result in data["_embedded"]["results"][
                            1:10
                        ]:  # Use more results if needed
                            if alt_result.get("title") != result["title"]:
                                result["similarOccupations"].append(
                                    {
                                        "title": alt_result.get("title", ""),
                                        "relation": "Similar role",
                                    }
                                )
                        return result

        # If we get here, the search failed or found no results
        return get_fallback_occupations(job_title)

    except Exception as e:
        print(f"Error in ESCO API search: {e}")
        return get_fallback_occupations(job_title)


def get_fallback_occupations(job_title):
    """Provide fallback occupation suggestions if ESCO API fails"""
    # Generic tech jobs as fallback
    tech_jobs = [
        {"title": "Software Developer", "relation": "Popular tech role"},
        {"title": "Web Developer", "relation": "Popular tech role"},
        {"title": "Data Analyst", "relation": "Popular tech role"},
        {"title": "Project Manager", "relation": "Popular tech role"},
        {"title": "UX Designer", "relation": "Popular tech role"},
    ]

    # Non-tech jobs as fallback
    non_tech_jobs = [
        {"title": "Marketing Specialist", "relation": "Popular role"},
        {"title": "Product Manager", "relation": "Popular role"},
        {"title": "Business Analyst", "relation": "Popular role"},
        {"title": "Operations Manager", "relation": "Popular role"},
        {"title": "HR Specialist", "relation": "Popular role"},
    ]

    # Try to infer if this is a tech job based on keywords
    tech_keywords = [
        "developer",
        "engineer",
        "programmer",
        "analyst",
        "technician",
        "designer",
        "architect",
        "admin",
        "specialist",
        "consultant",
    ]

    is_tech_job = any(keyword in job_title.lower() for keyword in tech_keywords)

    return {
        "title": job_title,
        "description": "No detailed description available.",
        "similarOccupations": tech_jobs if is_tech_job else non_tech_jobs,
    }


def search_esco_api(skill_name):
    """
    Search the ESCO API for information about a specific skill.

    Args:
        skill_name (str): The name of the skill to search for

    Returns:
        dict: Structured data about the skill from ESCO, or None if not found
    """
    url = "https://ec.europa.eu/esco/api/search"

    params = {
        "language": "en",
        "type": "skill",
        "text": skill_name,
        "selectedVersion": "v1.2.0",
        "full": "true",
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check if we have results
            if "_embedded" in data and "results" in data["_embedded"]:
                results = data["_embedded"]["results"]

                if results:
                    first_result = results[0]

                    # Extract useful information
                    skill_data = {
                        "title": first_result.get("title", ""),
                        "uri": first_result.get("uri", ""),
                        "description": first_result.get("description", {})
                        .get("en", {})
                        .get("literal", ""),
                    }

                    # Get skill type and reusability from _links
                    if (
                        "_links" in first_result
                        and "hasSkillType" in first_result["_links"]
                    ):
                        skill_types = first_result["_links"]["hasSkillType"]
                        if skill_types and len(skill_types) > 0:
                            skill_data["skillType"] = skill_types[0].get("title", "")

                    # Extract broader and narrower skills from _links
                    broader_skills = []
                    narrower_skills = []

                    if (
                        "_links" in first_result
                        and "broaderSkill" in first_result["_links"]
                    ):
                        for skill in first_result["_links"]["broaderSkill"]:
                            if "title" in skill:
                                broader_skills.append(skill["title"])

                    if (
                        "_links" in first_result
                        and "narrowerSkill" in first_result["_links"]
                    ):
                        for skill in first_result["_links"]["narrowerSkill"]:
                            if "title" in skill:
                                narrower_skills.append(skill["title"])

                    skill_data["broaderSkills"] = broader_skills
                    skill_data["narrowerSkills"] = narrower_skills

                    # Extract occupations that use this skill
                    occupations = []
                    if "_links" in first_result:
                        if "isEssentialForOccupation" in first_result["_links"]:
                            for occ in first_result["_links"][
                                "isEssentialForOccupation"
                            ]:
                                if "title" in occ:
                                    occupations.append(f"{occ['title']} (Essential)")

                        if "isOptionalForOccupation" in first_result["_links"]:
                            for occ in first_result["_links"][
                                "isOptionalForOccupation"
                            ]:
                                if "title" in occ:
                                    occupations.append(f"{occ['title']} (Optional)")

                    skill_data["occupations"] = occupations

                    return skill_data

            # No results found
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling ESCO API: {e}")
        return None


def list_esco_matches(skill_name, size=50):
    """
    Return a list of all ESCO search hits for `skill_name`.
    Each hit is the raw dict from the API, containing 'title', 'uri', etc.
    """
    url = "https://ec.europa.eu/esco/api/search"
    params = {
        "language": "en",
        "type": "skill",
        "text": skill_name,
        "selectedVersion": "v1.2.0",
        "full": "false",  # we only need labels + uri here
        "size": size,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("_embedded", {}).get("results", [])


def getLinks(course, provider, show_debug_in_main_thread=False):
    final_link, search_method = (
        search_course_link_serpapi(course, provider, SERPAPI_API_KEY),
        "SerpAPI",
    )
    if not final_link:
        search_method = "None"
    return final_link, search_method


def process_course_for_link(course_info):
    course = course_info["course"]
    provider = course_info["provider"]
    # Handle both "skills" array and legacy "skill" field
    skills = course_info.get("skills", [])
    if not skills and "skill" in course_info:
        skills = [course_info["skill"]]

    relevance_score = course_info.get("relevance_score", 0)
    difficulty = course_info.get("difficulty", "")
    duration = course_info.get("duration")
    resource_type = course_info.get("resource_type", "Course")

    generated_link, search_method = getLinks(
        course, provider, show_debug_in_main_thread=False
    )

    return {
        "skills": skills,
        "course": course,
        "provider": provider,
        "relevance_score": relevance_score,
        "difficulty": difficulty,
        "duration": duration,
        "resource_type": resource_type,
        "link": generated_link,
        "search_method": search_method,
    }


@st.cache_resource
def load_spacy_model(ontology_skills_labels):
    print("Loading or retrieving cached SpaCy model...")
    nlp = spacy.load("en_core_web_sm")

    if "entity_ruler" in nlp.pipe_names:
        print("Removing existing EntityRuler...")
        nlp.remove_pipe("entity_ruler")

    print(f"Adding EntityRuler with {len(ontology_skills_labels)} patterns.")
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

    patterns = []
    for skill_label in ontology_skills_labels:
        patterns.append({"label": "SKILL", "pattern": skill_label.lower()})
        patterns.append({"label": "SKILL", "pattern": skill_label})
        if skill_label.lower() != skill_label.upper():
            patterns.append({"label": "SKILL", "pattern": skill_label.upper()})
        if skill_label.lower() != skill_label.title():
            patterns.append({"label": "SKILL", "pattern": skill_label.title()})

    ruler.add_patterns(patterns)

    print(f"SpaCy pipeline after adding EntityRuler: {nlp.pipe_names}")
    print(f"EntityRuler patterns count: {len(ruler.patterns)}")
    return nlp


def expand_skills_with_ontology(
    skills_list, graph, ontology_skills_map, onto_namespace_uri
):
    if not skills_list:
        return []

    initial_sanitised_lower = set(s.lower() for s in skills_list)
    expanded_sanitised_lower = set(initial_sanitised_lower)
    processed_uris = set()

    ONTO = Namespace(onto_namespace_uri)
    RELATED_SKILL_PROP = ONTO.relatedSkill

    print(f"Expanding skills. Initial: {initial_sanitised_lower}")

    # Create a queue for URIs to process (start with URIs from initial skills)
    uris_to_process = []
    for skill_str_lower in initial_sanitised_lower:
        uri = ontology_skills_map.get(skill_str_lower)
        if uri and uri not in processed_uris:
            uris_to_process.append(uri)
            processed_uris.add(uri)  # Mark as processed immediately

    # Process URIs - simple one-level expansion for now
    # (Could be made recursive for deeper inference if needed)
    while uris_to_process:
        current_uri = uris_to_process.pop(0)

        # Query for skills related to the current URI
        related_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX onto: <{onto_namespace_uri}>

            SELECT ?relatedLabel WHERE {{
                # Find skills that the current skill is related to OR that are related to the current skill
                {{ <{current_uri}> onto:relatedSkill ?relatedUri . }}
                UNION
                {{ ?relatedUri onto:relatedSkill <{current_uri}> . }}
                # Add other relationships here if desired (e.g., skos:broader)
                # UNION {{ <{current_uri}> skos:broader ?relatedUri . }} # Example: Infer broader skills

                ?relatedUri rdfs:label ?relatedLabel .
                FILTER(?relatedUri != <{current_uri}>) # Don't add the skill itself
            }}
        """
        try:
            results = graph.query(related_query)
            for row in results:
                related_label = str(row.relatedLabel)
                sanitised_related_lower = sanitise_skill(related_label).lower()

                if sanitised_related_lower not in expanded_sanitised_lower:
                    expanded_sanitised_lower.add(sanitised_related_lower)
                    print(
                        f"  Inferred skill '{sanitised_related_lower}' from relation to {current_uri}"
                    )

                    # multi level inference
                    related_uri_new = ontology_skills_map.get(sanitised_related_lower)
                    if related_uri_new and related_uri_new not in processed_uris:
                        uris_to_process.append(related_uri_new)
                        processed_uris.add(related_uri_new)

        except Exception as e:
            print(f"Warning: Error querying related skills for {current_uri}: {e}")

    print(f"Expansion complete. Final skills: {expanded_sanitised_lower}")
    return list(expanded_sanitised_lower)


def extract_skills_keyword(text, nlp_model):
    print(f"Extracting skills using keyword method from text: '{text[:50]}...'")
    doc = nlp_model(text)
    skills = {ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"}
    return [sanitise_skill(skill) for skill in skills]


@st.cache_resource
def load_skill_extraction_pipeline():
    print("Loading or retrieving cached LinkedIn NER model...")
    return pipeline(
        "ner",
        model="algiraldohe/lm-ner-linkedin-skills-recognition",
        aggregation_strategy="simple",
    )


def get_skills_with_most_resources(graph, limit=5):
    """
    Queries the ontology to find skills associated with the most learning resources.

    Args:
        graph (rdflib.Graph): The loaded ontology graph.
        limit (int): The maximum number of skills to return.

    Returns:
        list: A list of skill labels (strings) ordered by resource count descending.
    """
    # Ensure limit is an integer
    try:
        limit_int = int(limit)
    except (ValueError, TypeError):
        print("Warning: Invalid limit value provided. Defaulting to 5.")
        limit_int = 5

    # Use an f-string to embed the limit directly into the query
    query_string = f"""
        PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?skillLabel (COUNT(?resource) AS ?resourceCount)
        WHERE {{
          # Ensure we are counting actual Learning_Resource instances
          ?resource rdf:type onto:Learning_Resource .
          ?resource onto:skillTaught ?skill .
          # Ensure the skill itself is defined as a Skill (optional but good practice)
          # ?skill rdf:type onto:Skill .
          ?skill rdfs:label ?skillLabel .
        }}
        GROUP BY ?skillLabel
        ORDER BY DESC(?resourceCount)
        LIMIT {limit_int}
    """

    try:
        results = graph.query(query_string)
        top_skills = [str(row.skillLabel) for row in results]
        print(f"Found top skills by resource count: {top_skills}")
        return top_skills
    except Exception as e:
        st.error(f"Error querying for popular skills: {e}")
        print(f"Error querying for popular skills: {e}")
        print("--- Query causing error ---")
        print(query_string)
        print("-------------------------")
        return []


def list_all_courses_in_ontology(graph):
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>
    SELECT ?res ?label ?skill WHERE {
        ?res rdf:type onto:Learning_Resource .
        ?res rdfs:label ?label .
        OPTIONAL { ?res onto:skillTaught ?skillUri . ?skillUri rdfs:label ?skill }
    }
    """
    results = list(graph.query(query))
    return results


def extract_skills_hf(text, skill_extraction_pipeline):
    try:
        print(f"Extracting skills using LinkedIn NER model from text: '{text[:50]}...'")
        results = skill_extraction_pipeline(text)
        skill_spans = {res.get("word", res.get("entity")) for res in results}
        skills = [sanitise_skill(skill) for skill in skill_spans]
        return skills
    except Exception as e:
        st.error(f"Error during skill extraction: {e}")
        print(f"Error during skill extraction: {e}")
        return []


def filter_valid_skills(extracted_skills, ontology_skills_sanitised):
    print(f"Filtering extracted skills: {extracted_skills}")
    valid_skills = set(skill.lower() for skill in ontology_skills_sanitised)
    matched_skills = []
    for extracted in extracted_skills:
        extracted_lower = extracted.lower()
        is_matched = False
        if extracted_lower in valid_skills:
            matched_skills.append(extracted)
            is_matched = True
        else:
            for valid in valid_skills:
                if extracted_lower in valid or valid in extracted_lower:
                    matched_skills.append(extracted)
                    is_matched = True
                    break
        if not is_matched and extracted:
            print(
                f"No direct or partial match found in ontology for extracted skill: {extracted}"
            )
    matched_skills = list(dict.fromkeys(matched_skills))
    return matched_skills


def filter_recommendations_by_type(
    recommendations, resource_type=None, difficulty=None
):
    """Filter course recommendations by resource type and difficulty"""
    if not (resource_type or difficulty):
        return recommendations

    filtered_recommendations = {}

    for skill, courses in recommendations.items():
        filtered_courses = []
        for course in courses:
            if (
                resource_type
                and course.get("resource_type", "").lower() != resource_type.lower()
            ):
                continue

            if (
                difficulty
                and course.get("difficulty", "").lower() != difficulty.lower()
            ):
                continue

            filtered_courses.append(course)

        if filtered_courses:
            filtered_recommendations[skill] = filtered_courses

    return filtered_recommendations


@st.cache_resource
def load_ontology():
    print("Loading or retrieving cached ontology...")
    ontology_iri = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology"
    onto = Namespace(ontology_iri + "#")
    data = Namespace(ontology_iri + "/upskilling#")
    g = Graph()
    try:
        g.parse("skill_ontology.ttl", format="turtle")
        print("Ontology file 'skill_ontology.ttl' parsed successfully.")
    except FileNotFoundError:
        st.error("Ontology file 'skill_ontology.ttl' not found!")
        print("Error: Ontology file 'skill_ontology.ttl' not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error parsing ontology file: {e}")
        print(f"Error parsing ontology file: {e}")
        st.stop()

    ontology_skills_labels = []
    ontology_skills_map = {}
    skill_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>

        SELECT ?skill ?label WHERE {
            ?skill rdf:type onto:Skill .
            { ?skill rdfs:label ?label . } # Get the primary label
            UNION
            { ?skill skos:altLabel ?label . } # Get alternative labels
        }
    """
    print(
        "Querying ontology for skill labels (rdfs:label and skos:altLabel) and URIs..."
    )
    for row in g.query(skill_query):
        uri, label = row
        if label:
            label_str = str(label)
            sanitised_label = sanitise_skill(label_str)
            # Add label to the list if you still need a flat list of all labels
            if label_str not in ontology_skills_labels:
                ontology_skills_labels.append(label_str)
            # Map the sanitised label (lowercase) to the skill URI
            ontology_skills_map[sanitised_label.lower()] = uri
            print(f"  Mapping '{sanitised_label.lower()}' -> {uri}")  # Debug print

    print(f"Found {len(ontology_skills_labels)} unique skill labels in ontology.")
    print(
        f"Created {len(ontology_skills_map)} sanitised skill URI mappings (incl. altLabels)."
    )

    return g, onto, data, ontology_skills_labels, ontology_skills_map


def calculate_course_relevance(course, skill):
    relevance_score = 0
    relevance_score += 1
    if skill.replace("_", " ").lower() in course["course"].lower():
        relevance_score += 3
    provider = course["provider"].lower()
    for top_provider in TOP_PROVIDERS:
        if top_provider in provider:
            relevance_score += 2
            break
    title_length = len(course["course"])
    if 15 <= title_length <= 70:
        relevance_score += 1
    if provider != "unknown":
        relevance_score += 1
    return relevance_score


def recommend_learning_resources(
    skills, graph, ontology_skills_map, max_courses_per_skill=3
):
    print(f"Finding learning resources for skills: {skills}")
    recommendations = {}
    all_courses_from_ontology = {}
    ONTO_URI = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#"
    ONTO = Namespace(ONTO_URI)

    def _find_courses_for_skill_uri(target_skill_uri, graph):
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX onto: <{ONTO_URI}>
            SELECT DISTINCT ?res ?label ?providerName ?rating ?difficulty ?duration ?resourceType ?taughtSkillLabel
            WHERE {{
                ?res rdf:type onto:Learning_Resource ;
                     rdfs:label ?label ;
                     onto:skillTaught <{target_skill_uri}> .
                <{target_skill_uri}> rdfs:label ?taughtSkillLabel .
                OPTIONAL {{ ?res onto:providedBy ?prov . ?prov rdfs:label ?providerName . }}
                OPTIONAL {{ ?res onto:hasRating ?rating . }}
                OPTIONAL {{ ?res onto:hasDifficultyLevel ?diffLevel . ?diffLevel rdfs:label ?difficulty . }}
                OPTIONAL {{ ?res onto:estimatedDuration ?duration . }}
                OPTIONAL {{ ?res onto:hasResourceType ?type . ?type rdfs:label ?resourceType . }}
            }} ORDER BY DESC(?rating) """
        try:
            return list(graph.query(query))
        except Exception as e:
            print(f"Error executing course query for URI {target_skill_uri}: {e}")
            return []

    for skill_sanitised in skills:
        skill_lower = skill_sanitised.lower()
        recommendations[skill_sanitised] = []
        all_courses_from_ontology[skill_sanitised] = []
        found_courses = False
        skill_uri = ontology_skills_map.get(skill_lower)
        print(f"Attempting lookup for '{skill_lower}'. URI found: {skill_uri}")

        if not skill_uri:
            print(
                f"Warning: No URI found for skill '{skill_sanitised}' in map. Skipping."
            )
            continue

        print(
            f"Searching direct resources for skill: {skill_sanitised} (URI: {skill_uri})"
        )
        direct_results = _find_courses_for_skill_uri(skill_uri, graph)

        if direct_results:
            print(f"Found {len(direct_results)} direct courses for {skill_sanitised}.")
            found_courses = True
            for res in direct_results:
                course_info = {
                    "course": str(res.label),
                    "provider": (
                        str(res.providerName)
                        if hasattr(res, "providerName") and res.providerName
                        else "Unknown"
                    ),
                    "rating": (
                        float(res.rating)
                        if hasattr(res, "rating") and res.rating
                        else None
                    ),
                    "difficulty": (
                        str(res.difficulty)
                        if hasattr(res, "difficulty") and res.difficulty
                        else None
                    ),
                    "duration": (
                        float(res.duration)
                        if hasattr(res, "duration") and res.duration
                        else None
                    ),
                    "resource_type": (
                        str(res.resourceType)
                        if hasattr(res, "resourceType") and res.resourceType
                        else "Course"
                    ),
                    "recommended_for_skill": skill_sanitised,
                    "actual_skill_taught": (
                        str(res.taughtSkillLabel)
                        if hasattr(res, "taughtSkillLabel")
                        else skill_sanitised
                    ),
                    "is_related_recommendation": False,
                }
                all_courses_from_ontology[skill_sanitised].append(course_info)
        else:
            print(
                f"No direct courses found for {skill_sanitised}. Checking RELATED (broader/narrower) skills..."
            )

            related_query = f"""
                PREFIX skos: <{SKOS}>
                PREFIX rdfs: <{RDFS}>
                SELECT DISTINCT ?relatedUri ?relatedLabel WHERE {{
                    {{ <{skill_uri}> skos:narrower ?relatedUri . }}
                    UNION
                    {{ <{skill_uri}> skos:broader ?relatedUri . }}
                    ?relatedUri rdfs:label ?relatedLabel .
                }}
                ORDER BY ?relatedLabel
            """
            try:
                related_skills = list(graph.query(related_query))
                if not related_skills:
                    print(
                        f"No broader or narrower skills found defined for {skill_sanitised}."
                    )

                for related_row in related_skills:
                    related_uri = related_row.relatedUri
                    related_label = str(related_row.relatedLabel)
                    print(
                        f"  Checking related skill: {related_label} (URI: {related_uri})"
                    )

                    related_results = _find_courses_for_skill_uri(related_uri, graph)

                    if related_results:
                        print(
                            f"  Found {len(related_results)} courses for related skill {related_label}."
                        )
                        found_courses = True
                        for res in related_results:
                            course_info = {
                                "course": str(res.label),
                                "provider": (
                                    str(res.providerName)
                                    if hasattr(res, "providerName") and res.providerName
                                    else "Unknown"
                                ),
                                "rating": (
                                    float(res.rating)
                                    if hasattr(res, "rating") and res.rating
                                    else None
                                ),
                                "difficulty": (
                                    str(res.difficulty)
                                    if hasattr(res, "difficulty") and res.difficulty
                                    else None
                                ),
                                "duration": (
                                    float(res.duration)
                                    if hasattr(res, "duration") and res.duration
                                    else None
                                ),
                                "resource_type": (
                                    str(res.resourceType)
                                    if hasattr(res, "resourceType") and res.resourceType
                                    else "Course"
                                ),
                                "recommended_for_skill": skill_sanitised,
                                "actual_skill_taught": related_label,
                                "is_related_recommendation": True,
                            }
                            if not any(
                                c["course"] == course_info["course"]
                                and c["provider"] == course_info["provider"]
                                for c in all_courses_from_ontology[skill_sanitised]
                            ):
                                all_courses_from_ontology[skill_sanitised].append(
                                    course_info
                                )
                        # break # Decide if you want courses from *all* related skills or just the first

            except Exception as e:
                print(f"Error querying related skills for {skill_uri}: {e}")

        if found_courses:
            courses_for_scoring = all_courses_from_ontology.get(skill_sanitised, [])
            for course in courses_for_scoring:
                score_skill = course["actual_skill_taught"]
                course["relevance_score"] = calculate_course_relevance(
                    course, score_skill
                )

            recommendations[skill_sanitised] = sorted(
                courses_for_scoring,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )
            recommendations[skill_sanitised] = recommendations[skill_sanitised][
                :max_courses_per_skill
            ]
        else:
            print(
                f"No courses found for '{skill_sanitised}' or its related (broader/narrower) skills."
            )

    return recommendations, all_courses_from_ontology


def get_learning_path_for_skill(skill_label, graph):
    """
    Retrieves a complete learning path for a given specific skill label.

    Args:
        skill_label (str): The exact label of the target skill.
        graph (Graph): The loaded ontology graph.

    Returns:
        dict or None: Path data including 'path_uri', 'path_label', 'resources',
                      or None if no path is found for this specific label.
    """
    # Use lowercase for matching robustness in FILTER
    skill_label_lower = skill_label.lower()

    # Query for paths targeting this specific skill label
    path_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>

        SELECT ?path ?pathLabel WHERE {{
          ?path rdf:type onto:LearningPath ;
                rdfs:label ?pathLabel ;
                onto:targetSkill/rdfs:label ?targetSkillLabel .
          FILTER(LCASE(STR(?targetSkillLabel)) = "{skill_label_lower}")
        }}
        LIMIT 1
        """
    try:
        paths = list(graph.query(path_query))
    except Exception as e:
        print(f"Error querying for direct path for '{skill_label}': {e}")
        return None

    if not paths:
        print(f"No direct learning path found for label: '{skill_label}'")
        return None  # No direct path found

    path_uri = paths[0][0]
    path_label_str = str(paths[0][1])
    print(f"Found direct path '{path_label_str}' for skill '{skill_label}'")

    # --- Resource Fetching Logic (Can be kept here or moved to helper) ---
    first_query = f"""
        PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>
        SELECT ?firstResource WHERE {{ <{path_uri}> onto:hasFirstResource ?firstResource . }}
        """
    first_result = list(graph.query(first_query))
    if not first_result:
        return None  # Path exists but has no first resource

    first_resource = first_result[0][0]
    resources = []
    current = first_resource
    processed_uris = set()  # Prevent infinite loops in hasNextResource cycles

    while current and current not in processed_uris:
        processed_uris.add(current)
        resource_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX onto: <http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#>
            SELECT ?label ?difficulty ?duration ?type ?provider ?rating ?next
            WHERE {{
              <{current}> rdfs:label ?label .
              OPTIONAL {{ <{current}> onto:hasDifficultyLevel/rdfs:label ?difficulty . }}
              OPTIONAL {{ <{current}> onto:estimatedDuration ?duration . }}
              OPTIONAL {{ <{current}> onto:hasResourceType/rdfs:label ?type . }}
              OPTIONAL {{ <{current}> onto:providedBy/rdfs:label ?provider . }}
              OPTIONAL {{ <{current}> onto:hasRating ?rating . }}
              OPTIONAL {{ <{current}> onto:hasNextResource ?next . }}
            }} LIMIT 1
            """
        result = list(graph.query(resource_query))
        if result:
            data = result[0]
            resources.append(
                {
                    "uri": str(current),
                    "title": str(data.label) if data.label else "Untitled Step",
                    "difficulty": (
                        str(data.difficulty) if data.difficulty else "Not specified"
                    ),
                    "duration": float(data.duration) if data.duration else None,
                    "type": str(data.type) if data.type else "Course",
                    "provider": str(data.provider) if data.provider else "Unknown",
                    "rating": float(data.rating) if data.rating else None,
                }
            )
            current = data.next if hasattr(data, "next") and data.next else None
        else:
            print(
                f"Warning: Could not fetch details for resource {current} in path {path_uri}"
            )
            break  # Stop if resource details can't be fetched

    return {
        "path_uri": str(path_uri),
        "path_label": path_label_str,
        "resources": resources,
    }


def _display_path_steps(path_data):
    """Helper function to render the steps of a learning path."""
    st.markdown(f"##### Path: {path_data['path_label']}")
    st.markdown("---")
    if not path_data["resources"]:
        st.warning("This learning path currently has no steps defined.")
        return

    for i, resource in enumerate(path_data["resources"]):
        st.markdown(f"**Step {i + 1}: {resource['title']}**")

        meta_info = []
        if resource["difficulty"] and resource["difficulty"] != "Not specified":
            # Ensure get_difficulty_badge is available
            meta_info.append(get_difficulty_badge(resource["difficulty"]))
        if resource["type"]:
            meta_info.append(f"Type: {resource['type']}")
        if resource["provider"] != "Unknown":
            meta_info.append(f"Provider: {resource['provider']}")
        if resource["duration"]:
            meta_info.append(f"Duration: {resource['duration']} hours")
        if resource["rating"]:
            meta_info.append(f"Rating: {resource['rating']}/5.0")

        if meta_info:
            st.markdown(" | ".join(meta_info), unsafe_allow_html=True)
        else:
            st.markdown("_(No further details specified)_")

        st.markdown("---")  # Separator after each step


def show_learning_path(skill_label, graph, ontology_skills_map, onto_namespace_uri):
    """
    Displays a learning path for the skill. If no direct path is found,
    searches for paths associated with related skills (via onto:relatedSkill).

    Args:
        skill_label (str): The label of the skill to find a path for.
        graph (Graph): The loaded ontology graph.
        ontology_skills_map (dict): Mapping from lowercased sanitised skill string to skill URI.
        onto_namespace_uri (str): The base URI string for the ontology.

    Returns:
        bool: True if a path (direct or related) was found and displayed, False otherwise.
    """
    ONTO = Namespace(onto_namespace_uri)

    # --- Try finding direct path first ---
    print(f"show_learning_path: Searching direct path for '{skill_label}'")
    direct_path_data = get_learning_path_for_skill(skill_label, graph)

    if direct_path_data:
        print("show_learning_path: Found direct path. Displaying.")
        _display_path_steps(direct_path_data)
        return True

    # --- If no direct path, try related skills ---
    print(
        f"show_learning_path: No direct path found for '{skill_label}'. Checking related skills."
    )
    # Find the URI of the original skill
    skill_sanitised_lower = sanitise_skill(skill_label).lower()
    skill_uri = ontology_skills_map.get(skill_sanitised_lower)

    if not skill_uri:
        print(
            f"show_learning_path: Cannot find URI for '{skill_label}' in map. Cannot search related."
        )
        st.info(
            f"No predefined learning path found for '{skill_label}'."
        )  # Default message
        return False  # Cannot proceed without URI

    # Query for related skills
    related_skills_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX onto: <{onto_namespace_uri}>
        SELECT DISTINCT ?relatedLabel ?relatedUri WHERE {{
            {{ <{skill_uri}> onto:relatedSkill ?relatedUri . }}
            UNION
            {{ ?relatedUri onto:relatedSkill <{skill_uri}> . }}
            ?relatedUri rdfs:label ?relatedLabel .
            FILTER(?relatedUri != <{skill_uri}>)
        }}
    """
    try:
        related_skills_results = graph.query(related_skills_query)
        found_related_path = False
        for row in related_skills_results:
            related_label = str(row.relatedLabel)
            print(
                f"show_learning_path: Checking related skill '{related_label}' for paths..."
            )
            # Try finding a path for the related skill's label
            related_path_data = get_learning_path_for_skill(related_label, graph)
            if related_path_data:
                print(
                    f"show_learning_path: Found path for related skill '{related_label}'. Displaying."
                )
                st.info(
                    f"No direct path found for '{skill_label}'. Showing path for related skill: **{related_label}**"
                )
                st.markdown("---")
                _display_path_steps(related_path_data)
                found_related_path = True
                break  # Stop after finding the first related path

        if found_related_path:
            return True

    except Exception as e:
        print(
            f"show_learning_path: Error querying/processing related skills for '{skill_label}': {e}"
        )
        # Fall through to the final message

    # --- If no direct or related path found ---
    print(f"show_learning_path: No direct or related paths found for '{skill_label}'.")
    st.info(
        f"No predefined learning path found for '{skill_label}' or its direct relations."
    )
    return False


def extract_skills_from_resume(
    resume_text,
    nlp_model,
    skill_extraction_pipeline,
    extraction_method,
    ontology_skills_sanitised,
):
    """Extract skills from resume text using the specified method"""
    print(f"Extracting skills from resume using {extraction_method} method")
    if extraction_method == "Keyword-based":
        extracted_skills = extract_skills_keyword(resume_text, nlp_model)
    else:
        extracted_skills = extract_skills_hf(resume_text, skill_extraction_pipeline)

    filtered_skills = filter_valid_skills(extracted_skills, ontology_skills_sanitised)
    return filtered_skills


def compare_skills(
    job_skills,
    effective_resume_skills_list,
    rejected_skills_set,
    graph,
    ontology_skills_map,
    onto_namespace_uri,
):
    ONTO = Namespace(onto_namespace_uri)
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    resume_set_lower = set(s.lower() for s in effective_resume_skills_list)
    job_skills_sanitised = [sanitise_skill(js) for js in job_skills]
    matching_skills = []
    missing_skills = []
    print(f"Compare Check - Job Skills (Sanitised): {job_skills_sanitised}")
    print(f"Compare Check - Resume Effective (Lower, Sanitised): {resume_set_lower}")
    print(f"Compare Check - Rejected Skills Set: {rejected_skills_set}")
    for job_skill_sanitised in job_skills_sanitised:
        job_skill_lower = job_skill_sanitised.lower()
        is_matched = False
        if job_skill_lower in resume_set_lower:
            if job_skill_lower not in rejected_skills_set:
                is_matched = True
                print(f"  Direct match found for: {job_skill_lower}")
            else:
                print(
                    f"  Skipping direct match for '{job_skill_lower}' as it was rejected."
                )
        if not is_matched and job_skill_lower not in rejected_skills_set:
            job_skill_uri = ontology_skills_map.get(job_skill_lower)
            if job_skill_uri:
                broader_query = f"""\
PREFIX skos: <{SKOS}>
SELECT ?broaderSkillUri WHERE {{
    <{job_skill_uri}> skos:broader ?broaderSkillUri .
}}
"""
                try:
                    results = graph.query(broader_query)
                    for row in results:
                        broader_uri = row.broaderSkillUri
                        label_query = f"SELECT ?label WHERE {{ <{broader_uri}> rdfs:label|skos:altLabel ?label . }}"
                        labels = graph.query(label_query)
                        for label_row in labels:
                            broader_label_sanitised_lower = sanitise_skill(
                                str(label_row.label)
                            ).lower()
                            if broader_label_sanitised_lower in resume_set_lower:
                                if (
                                    broader_label_sanitised_lower
                                    not in rejected_skills_set
                                ):
                                    print(
                                        f"  Match found for '{job_skill_lower}' via broader skill '{broader_label_sanitised_lower}'"
                                    )
                                    is_matched = True
                                    break
                        if is_matched:
                            break
                except Exception as e:
                    print(
                        f"Warning: Error querying broader skills for {job_skill_uri}: {e}"
                    )
            else:
                print(
                    f"  Warning: Cannot find URI for job skill '{job_skill_lower}' in map. Cannot check broader."
                )
        elif job_skill_lower in rejected_skills_set:
            print(
                f"  Skipping broader check for '{job_skill_lower}' because it was explicitly rejected."
            )
        if is_matched:
            matching_skills.append(job_skill_sanitised)
        else:
            missing_skills.append(job_skill_sanitised)
    return list(set(matching_skills)), list(set(missing_skills))


def display_related_skills(graph, skill_uri, onto_namespace):
    """Queries and displays skills related via onto:relatedSkill."""
    st.markdown("### Related Skills (from Ontology)")
    related_skills_query = f"""
        PREFIX onto: <{onto_namespace}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?relatedSkillLabel ?relatedSkillUri
        WHERE {{
            {{ <{skill_uri}> onto:relatedSkill ?relatedSkillUri . }}
            UNION
            {{ ?relatedSkillUri onto:relatedSkill <{skill_uri}> . }}
            ?relatedSkillUri rdfs:label ?relatedSkillLabel .
            FILTER(?relatedSkillUri != <{skill_uri}>)
        }}
        ORDER BY ?relatedSkillLabel
    """
    try:
        related_results = list(graph.query(related_skills_query))
        if related_results:
            st.write("This skill is related to:")
            for row in related_results:
                related_label = str(row.relatedSkillLabel)
                sanitised_related = sanitise_skill(related_label)
                link = f"?page=skills&skill={sanitised_related}"
                st.markdown(f"- [{related_label}]({link})")
        else:
            st.info("No related skills found in the ontology for this skill.")
    except Exception as e:
        st.error(f"Error querying for related skills: {e}")
        print(f"Error querying related skills for {skill_uri}: {e}")


def create_navbar():
    navbar_css = """
    <style>
        body, html {
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden;
        }

        .stApp {
             margin-top: 0 !important;
             padding-top: 0 !important;
         }

        .stApp [data-testid="stAppViewContainer"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
            max-width: unset !important;
            margin: 0 !important;
        }

        .stApp [data-testid="stHeader"] {
            display: none !important;
        }

        .stApp [data-testid="stSidebar"] {
            margin-top: 0 !important;
            padding-top: 55px !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }

        .navbar-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
            z-index: 1000;
            padding: 0;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 40px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0;
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            height: 100%;
            background-color: #000000;
            padding: 0 15px;
            color: white !important;
            text-decoration: none !important;
        }

        .navbar-brand img {
            height: 28px;
            width: auto;
        }

         .navbar-brand span {
            color: white !important;
            font-size: 1rem;
            font-weight: 700;
            margin-left: 10px;
         }

        .navbar-links {
            display: flex;
            height: 100%;
            align-items: center;
        }

        .navbar-page-link {
            color: white !important;
            text-decoration: none !important;
            font-size: 0.9rem;
            font-weight: 400;
            padding: 0 12px;
            display: flex;
            align-items: center;
            height: 100%;
            border-left: 1px solid #333;
            background: none !important;
            transition: background-color 0.2s ease;
        }

        .navbar-page-link:first-child {
             border-left: none;
        }

        .navbar-page-link:hover {
            background-color: #333 !important;
            color: white !important;
            text-decoration: none !important;
        }

        #MainMenu, footer, header {
            visibility: hidden !important;
            display: none !important;
        }

        .stDeployButton {
            display: none !important;
        }

        .main-content {
            margin-top: 60px;
            padding: 1rem;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }

        .footer {
            position: relative;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #333;
            text-align: center;
            color: #888;
            font-size: 0.8rem;
            width: 100%;
        }
    </style>
    """

    st.markdown(navbar_css, unsafe_allow_html=True)

    def img_to_base64(img_path):
        try:
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception as e:
            print(f"Could not load logo: {e}")
            return ""

    logo_base64 = img_to_base64("assets/img/logo.jpg")

    navbar_html = f"""
    <div class="navbar-wrapper">
        <div class="navbar">
            <a href="?page=home" target="_self" class="navbar-brand">
                <img src="data:image/png;base64,{logo_base64}" alt="RoleMate AI Logo">
                <span>RoleMate AI</span>
            </a>
            <div class="navbar-links">
                <a href="?page=home" target="_self" class="navbar-page-link">Home</a>
                <a href="?page=recommendations" target="_self" class="navbar-page-link">Recommendations</a>
                <a href="?page=skills" target="_self" class="navbar-page-link">Skills</a>
                <a href="?page=courses" target="_self" class="navbar-page-link">Courses</a>
                <a href="?page=progress" target="_self" class="navbar-page-link">My Progress</a>
            </div>
        </div>
    </div>
    """

    st.markdown(navbar_html, unsafe_allow_html=True)


def home_page():
    # Hero section
    st.markdown(
        """
<div style='background: url("https://imgur.com/a/3VSe1Eq") no-repeat center/cover; padding: 4rem 2rem; color: white; text-align: center;'>
        <h1 style='font-size:3rem; margin-bottom:0.5rem;'>Upskill Smarter with RoleMate AI</h1>
        <p style='font-size:1.2rem; margin-bottom:1.5rem;'>Paste a job description and get tailored skill extractions and course recommendations in seconds.</p>
        <a href='?page=recommendations' target='_self' style='background:#ff6600;color:white;padding:0.8rem 1.5rem;border-radius:4px;text-decoration:none;font-weight:bold;'>Get Started</a>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("assets/svg/undraw_text-files_tqjw.svg", width=80)
        st.markdown(
            "**1. Analyse Job Requirements**  \n"
            "Paste any job description or upload a PDF resume to identify required skills."
        )

    with col2:
        st.image("assets/svg/undraw_file-search_cbur.svg", width=80)
        st.markdown(
            "**2. Skills Assessment**  \n"
            "Our AI identifies key skills from both your resume and job descriptions."
        )

    with col3:
        st.image(
            "assets/svg/undraw_books_wxzz.svg", width=80
        ) 
        st.markdown(
            "**3. Personalised Learning**  \n"
            "Get tailored course recommendations and structured learning paths."
        )

    st.markdown("---")

    # Key metrics
    try:
        g, onto, data, ontology_skills_labels, ontology_skills_map = load_ontology()
        num_skills = len(ontology_skills_labels)
    except Exception:
        num_skills = "-"
    num_providers = len(TOP_PROVIDERS)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Skills in Database", num_skills)
    with col2:
        st.metric("Top Providers", num_providers)

    st.markdown("---")

    def render_skill_tags(skills):
        tags_html = "".join(
            f'<a href="?page=skills&skill={sanitise_skill(s)}" class="pill">{s}</a>'
            for s in skills
        )
        st.markdown(
            """
            <style>
            .pill {
                display: inline-block;
                background-color: #2c3e50;
                color: white !important;
                padding: 6px 12px;
                margin: 4px;
                border-radius: 12px;
                text-decoration: none;
                font-size: 0.9rem;
            }
            .pill:hover {
                background-color: #34495e;
            }
            </style>
            """
            + tags_html,
            unsafe_allow_html=True,
        )

    st.subheader("Popular Skills (Most Resources)")

    try:
        g, _, _, _, _ = load_ontology()

        popular_skills = get_skills_with_most_resources(g, limit=5)

        if popular_skills:
            render_skill_tags(popular_skills)
        else:
            st.info("Could not determine popular skills at this time.")

    except Exception as e:
        st.error(f"An error occurred while loading popular skills: {e}")

    st.markdown("---")

    st.subheader("FAQ")

    faq = {
        "📄 What’s supported?": "We currently support plain-text job descriptions ans PDF uploads. DOCX parsing is coming soon!",
        "🔍 How are courses chosen?": (
            "We map each missing skill to courses in our ontology and pull a fresh link via SerpAPI, "
            "then show  the highest-scoring options based on provider reputation, course relevance, and learner ratings."
        ),
        "🛠️ Can I request new skills?": (
            "Absolutely—open an issue on our "
            "[GitHub](https://github.com/yourrepo) and we’ll review and add it."
        ),
    }

    for question, answer in faq.items():
        with st.expander(question):
            st.markdown(answer)


# ──────────────────────────────────────────────────────────────────────────────
#  Recommendations Page
# ──────────────────────────────────────────────────────────────────────────────
def recommendations_page():
    st.markdown(
        "<h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>Upskilling Recommendations</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # --- Tabs ---
    job_tab, resume_tab, results_tab = st.tabs(["Job Description", "Resume", "Results"])

    # --- Session defaults ---
    defaults = {
        "job_skills": [],
        "resume_skills": [],
        "resume_skills_expanded": [],
        "rejected_inferred_skills": set(),
        "analysis_complete": False,
        "max_courses": 3,
        "show_debug": False,
        "ontology_graph": None,
        "ontology_skills_map": None,
        "ontology_skills_labels": [],
        "job_title": "",
        "recommendations_generated": False,
        "processed_recommendations": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # --- Job Description Tab ---
    with job_tab:
        st.markdown("Enter a job title and description to get started…")
        job_title = st.text_input(
            "Job Title",
            value=st.session_state.job_title,
            placeholder="e.g., Data Scientist",
            key="job_title_input",
        )
        job_desc = st.text_area("Job Description", height=200, key="job_desc_input")
        st.markdown("Elevate your skills with our AI-powered extraction! ✨")
        st.session_state.show_debug = st.checkbox(
            "Show Debug Information",
            value=st.session_state.show_debug,
            key="job_debug_cb",
        )

        if st.button("Extract Skills from Job", key="extract_job_btn"):
            if not job_desc:
                st.warning("Please enter a job description to proceed.")
                st.stop()

            with st.spinner("Analysing job description…"):
                # --- Load Ontology if needed ---
                if (
                    st.session_state.ontology_graph is None
                    or st.session_state.ontology_skills_map is None
                ):
                    print("Loading ontology for Job analysis...")
                    try:
                        # Ensure load_ontology returns all needed parts
                        g_job, _, _, labels_original_job, onto_map_job = load_ontology()
                        st.session_state.ontology_graph = g_job
                        st.session_state.ontology_skills_map = onto_map_job
                        st.session_state.ontology_skills_labels = (
                            labels_original_job  # Store labels too
                        )
                        print("Ontology loaded and stored in session state.")
                    except Exception as e:
                        st.error(f"Failed to load ontology: {e}")
                        st.stop()
                else:
                    print("Using ontology graph/map from session state for Job.")

                # Ensure pipeline is loaded
                if "skill_extraction_pipeline" not in st.session_state:
                    with st.spinner("Loading Hugging Face model…"):
                        st.session_state.skill_extraction_pipeline = (
                            load_skill_extraction_pipeline()
                        )
                pipe = st.session_state.skill_extraction_pipeline

                # Extract and filter
                extracted = extract_skills_hf(job_desc, pipe)
                # Use map keys (sanitised, lower) for filtering comparison
                onto_labels_sanitised_job = list(
                    st.session_state.ontology_skills_map.keys()
                )
                filtered = filter_valid_skills(extracted, onto_labels_sanitised_job)

            # Save results
            st.session_state.job_skills = filtered  # Store sanitised job skills
            st.session_state.job_title = job_title  # Update job title in session state

            st.success("Job skills extracted successfully!")
            if filtered:
                st.subheader("Extracted Job Skills")
                for sk in filtered:
                    # Attempt to find original label for display if possible, otherwise use sanitised
                    original_label = sk  # Default
                    for label in st.session_state.ontology_skills_labels:
                        if sanitise_skill(label).lower() == sk.lower():
                            original_label = label
                            break
                    st.write(f"• {original_label.replace('_', ' ').title()}")
            else:
                st.info("No skills matched the ontology in the job description.")

            # Reset analysis complete flag if job skills are re-extracted
            st.session_state.analysis_complete = False
            st.session_state.recommendations_generated = False
            st.session_state.processed_recommendations = []

            st.info("Now switch to the **Resume** tab to upload your resume.")

    # --- Resume Tab ---
    with resume_tab:
        print(
            f"DEBUG: Start of results_tab. Rejected set: {st.session_state.get('rejected_inferred_skills', 'Not Initialized Yet')}"
        )  # Add this
        st.markdown("Upload your resume or paste its content.")
        upload_option = st.radio(
            "Input method", ["Upload PDF", "Paste Text"], key="resume_input_option"
        )
        resume_text = ""

        if upload_option == "Upload PDF":
            uploaded = st.file_uploader(
                "Resume (PDF)", type="pdf", key="resume_pdf_upload"
            )
            if uploaded:
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded)
                    extracted_pages = [
                        p.extract_text() for p in pdf_reader.pages if p.extract_text()
                    ]
                    if not extracted_pages:
                        st.warning(
                            "Could not extract text from the PDF. Try pasting the text."
                        )
                        resume_text = ""
                    else:
                        resume_text = " ".join(extracted_pages)
                        st.success("PDF processed. Preview below:")
                        st.text_area(
                            "Extracted Text",
                            resume_text[:500] + "…",
                            height=150,
                            key="resume_pdf_preview",
                            disabled=True,
                        )
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    resume_text = ""  # Reset text on error
        else:
            resume_text = st.text_area(
                "Paste Resume Text", height=250, key="resume_text_paste"
            )

        if st.button("Extract Skills from Resume", key="extract_resume_btn"):
            if not resume_text:
                st.warning("Please supply resume text or upload a PDF.")
                st.stop()

            with st.spinner("Analysing resume…"):
                # --- Load Ontology if needed ---
                if (
                    st.session_state.ontology_graph is None
                    or st.session_state.ontology_skills_map is None
                ):
                    print("Loading ontology for Resume analysis...")
                    try:
                        g_resume, _, _, labels_original_resume, onto_map_resume = (
                            load_ontology()
                        )
                        st.session_state.ontology_graph = g_resume
                        st.session_state.ontology_skills_map = onto_map_resume
                        st.session_state.ontology_skills_labels = labels_original_resume
                        print("Ontology loaded and stored in session state.")
                    except Exception as e:
                        st.error(f"Failed to load ontology: {e}")
                        st.stop()
                else:
                    print("Using ontology graph/map from session state for Resume.")

                # Ensure pipeline is loaded
                if "skill_extraction_pipeline" not in st.session_state:
                    st.session_state.skill_extraction_pipeline = (
                        load_skill_extraction_pipeline()
                    )
                pipe = st.session_state.skill_extraction_pipeline

                # Extract and filter skills (get initial list)
                extracted = extract_skills_hf(resume_text, pipe)
                onto_labels_sanitised_resume = list(
                    st.session_state.ontology_skills_map.keys()
                )
                filtered = filter_valid_skills(extracted, onto_labels_sanitised_resume)

            # Store the *initial* filtered skills (sanitised strings)
            st.session_state.resume_skills = filtered
            st.success("Resume skills extracted successfully!")

            if filtered:
                st.subheader("Extracted Skills")
                for sk in filtered:
                    # Attempt to find original label for display
                    original_label = sk
                    for label in st.session_state.ontology_skills_labels:
                        if sanitise_skill(label).lower() == sk.lower():
                            original_label = label
                            break
                    st.write(f"• {original_label.replace('_', ' ').title()}")

                # --- Expand Resume Skills using Ontology ---
                with st.spinner(
                    "Checking for related skills in ontology..."
                ):  # Keep spinner for feedback
                    try:
                        ONTO_URI = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#"
                        if (
                            st.session_state.ontology_graph is None
                            or st.session_state.ontology_skills_map is None
                        ):
                            st.error(
                                "Ontology data is not available for skill expansion."
                            )
                            st.stop()

                        expanded_resume_skills_list = expand_skills_with_ontology(
                            st.session_state.resume_skills,
                            st.session_state.ontology_graph,
                            st.session_state.ontology_skills_map,
                            ONTO_URI,
                        )
                        st.session_state.resume_skills_expanded = (
                            expanded_resume_skills_list
                        )

                    except Exception as e:
                        st.error(f"Failed to expand resume skills using ontology: {e}")
                        print(f"Error expanding resume skills: {e}")
                        # Fallback: use the original filtered list (lowercase) if expansion fails
                        st.session_state.resume_skills_expanded = [
                            s.lower() for s in st.session_state.resume_skills
                        ]

            else:
                st.info("No skills were extracted from your resume.")
                st.session_state.resume_skills = []  # Ensure initial list is empty
                st.session_state.resume_skills_expanded = []  # Ensure expanded list is empty

            # Mark analysis complete only if job skills also exist
            st.session_state.analysis_complete = bool(st.session_state.job_skills)
            # Reset course generation flag when resume skills change
            st.session_state.recommendations_generated = False
            st.session_state.processed_recommendations = []

            if st.session_state.analysis_complete:
                st.info("Great! Head over to the **Results** tab.")
            else:
                st.warning("Please extract job skills first on the Job tab.")

    # --- Results Tab ---
    with results_tab:
        if not st.session_state.analysis_complete:
            st.info(
                "Please extract skills from both Job Description and Resume tabs first."
            )
            st.stop()
        if not st.session_state.job_skills:
            st.warning("Job skills have not been extracted yet.")
            st.stop()
        if (
            st.session_state.ontology_graph is None
            or st.session_state.ontology_skills_map is None
        ):
            st.error("Ontology data not loaded. Cannot perform skill comparison.")
            st.stop()

        st.subheader("📊 Skills Analysis Results")

        if "rejected_inferred_skills" not in st.session_state:
            st.session_state.rejected_inferred_skills = set()

        expanded_set = set(st.session_state.resume_skills_expanded)
        rejected_skills_set = st.session_state.rejected_inferred_skills
        effective_resume_skills_set = expanded_set - rejected_skills_set
        effective_resume_skills_list = list(effective_resume_skills_set)

        g_compare = st.session_state.ontology_graph
        map_compare = st.session_state.ontology_skills_map
        ONTO_URI = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#"

        matching, missing = compare_skills(
            st.session_state.job_skills,
            effective_resume_skills_list,
            rejected_skills_set,
            g_compare,
            map_compare,
            ONTO_URI,
        )

        # --- Calculate and Display Metrics ---
        unique_matching_count = len(set(matching))
        unique_missing_count = len(set(missing))
        unique_job_skills_count = len(set(st.session_state.job_skills))
        match_pct = (
            (unique_matching_count / unique_job_skills_count * 100)
            if unique_job_skills_count
            else 0
        )

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Match Score", f"{match_pct:.1f}%")
        with summary_cols[1]:
            st.metric("✅ Skills Matched", unique_matching_count)
        with summary_cols[2]:
            st.metric("🎯 Skills to Develop", unique_missing_count)
        st.markdown("---")

        # --- Display Matched and Missing Skills Lists (with interaction buttons) ---
        st.markdown("#### Skill Comparison Details")
        col_match_list, col_miss_list = st.columns(2)

        with col_match_list:
            st.markdown("##### ✅ Skills Matched")
            with st.container(border=True, height=250):
                if matching:
                    display_list = sorted(list(set(matching)))
                    for skill_matched_sanitised in display_list:
                        # Determine if it was inferred
                        is_inferred = skill_matched_sanitised.lower() not in (
                            s.lower() for s in st.session_state.resume_skills
                        )
                        # Check if it has been rejected (it shouldn't be in 'matching' if rejected, but double-check)
                        is_rejected = (
                            skill_matched_sanitised.lower()
                            in st.session_state.rejected_inferred_skills
                        )

                        # Find original label for display
                        original_label = skill_matched_sanitised
                        if "ontology_skills_labels" in st.session_state:
                            for label in st.session_state.ontology_skills_labels:
                                if (
                                    sanitise_skill(label).lower()
                                    == skill_matched_sanitised.lower()
                                ):
                                    original_label = label
                                    break
                        display_name = original_label.replace("_", " ").title()

                        # --- Add Buttons ---
                        cols = st.columns([0.8, 0.2])  # Adjust ratio as needed
                        with cols[0]:
                            # Display skill name and inferred status
                            st.markdown(
                                f"- {display_name} {'(inferred)' if is_inferred else ''}",
                                unsafe_allow_html=True,
                            )
                        with cols[1]:
                            # Show reject button ONLY if it's inferred AND NOT already rejected
                            if is_inferred and not is_rejected:
                                reject_key = f"reject_{skill_matched_sanitised.lower()}"
                                if st.button(
                                    "❌",
                                    key=reject_key,
                                    help=f"Reject inferred skill '{display_name}'",
                                ):
                                    st.session_state.rejected_inferred_skills.add(
                                        skill_matched_sanitised.lower()
                                    )
                                    print(
                                        f"User rejected: {skill_matched_sanitised.lower()}"
                                    )
                                    st.rerun()  # Rerun immediately to update lists

                else:
                    st.info("No matching skills found.")

        with col_miss_list:
            st.markdown("##### 🎯 Skills to Develop (Gap)")
            with st.container(border=True, height=250):
                if missing:
                    display_list_missing = sorted(list(set(missing)))
                    for skill_sanitised in display_list_missing:
                        skill_lower = (
                            skill_sanitised.lower()
                        )  # Use lowercase for checks

                        original_label = skill_sanitised
                        if "ontology_skills_labels" in st.session_state:
                            for label in st.session_state.ontology_skills_labels:
                                if sanitise_skill(label).lower() == skill_lower:
                                    original_label = label
                                    break
                        display_name = original_label.replace("_", " ").title()

                        # Check if this missing skill was originally inferred but then rejected
                        was_inferred = skill_lower not in (
                            s.lower() for s in st.session_state.resume_skills
                        )
                        is_rejected = (
                            skill_lower in st.session_state.rejected_inferred_skills
                        )

                        item_cols = st.columns([0.8, 0.2])
                        with item_cols[0]:
                            st.markdown(f"- {display_name}", unsafe_allow_html=True)
                        with item_cols[1]:
                            # Show accept/undo button ONLY if it was inferred AND is currently rejected
                            if was_inferred and is_rejected:
                                accept_key = f"accept_{skill_lower}"  # Use lower for key consistency
                                if st.button(
                                    "✅",
                                    key=accept_key,
                                    help=f"Accept inferred skill '{display_name}' again",
                                ):
                                    st.session_state.rejected_inferred_skills.discard(
                                        skill_lower
                                    )
                                    print(
                                        f"User accepted: {skill_lower}. New rejected set: {st.session_state.rejected_inferred_skills}"
                                    )
                                    st.rerun()  # Trigger rerun
                else:
                    st.success("Excellent! No skill gaps found.")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(
            ["🎯 Skill Gap Details", "💼 Similar Roles", "📚 All Recommended Courses"]
        )

        with tab1:
            if not missing:
                st.success(
                    "Excellent! No skill gaps found, so no specific details to show here."
                )
            else:
                unique_missing_list = sorted(list(set(missing)))
                st.markdown(
                    f"Focus on developing these **{len(unique_missing_list)}** skill(s):"
                )
                try:
                    g_details = st.session_state.ontology_graph
                    map_details = st.session_state.ontology_skills_map
                    graph_loaded_details = True
                    initial_recs_data, _ = recommend_learning_resources(
                        unique_missing_list,
                        g_details,
                        map_details,
                        max_courses_per_skill=5,
                    )
                except Exception as e:
                    st.error(f"Failed to load initial courses for details tab: {e}")
                    graph_loaded_details = False
                    initial_recs_data = {}

                for skill_sanitised in unique_missing_list:
                    original_label = skill_sanitised
                    if "ontology_skills_labels" in st.session_state:
                        for label in st.session_state.ontology_skills_labels:
                            if sanitise_skill(label).lower() == skill_sanitised.lower():
                                original_label = label
                                break
                    skill_title = original_label.replace("_", " ").title()

                    with st.expander(f"🎯 {skill_title}"):
                        st.markdown("##### 🗺️ Suggested Learning Path")
                        if graph_loaded_details:
                            ONTO_URI_LP = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#"
                            path_found = show_learning_path(
                                original_label, g_details, map_details, ONTO_URI_LP
                            )
                        else:
                            st.warning(
                                "Learning path display unavailable (ontology error)."
                            )
                        st.markdown("---")

                        st.markdown(f"##### 📚 Top Courses for {skill_title}")
                        if graph_loaded_details:
                            courses_for_this_skill = initial_recs_data.get(
                                skill_sanitised, []
                            )
                            if courses_for_this_skill:
                                for i, course_data in enumerate(
                                    courses_for_this_skill[:3]
                                ):
                                    st.markdown(f"**{i + 1}. {course_data['course']}**")
                                    meta = []
                                    if course_data.get("provider") != "Unknown":
                                        meta.append(f"🎓 {course_data['provider']}")
                                    if course_data.get("difficulty"):
                                        meta.append(
                                            f"{get_difficulty_badge(course_data['difficulty'])}"
                                        )
                                    if course_data.get("resource_type"):
                                        meta.append(
                                            f"📘 {course_data['resource_type']}"
                                        )
                                    st.markdown(
                                        " | ".join(meta), unsafe_allow_html=True
                                    )
                                    st.markdown(
                                        "<small><i>(Full details & link in 'All Recommended Courses' tab)</i></small>",
                                        unsafe_allow_html=True,
                                    )
                                    if i < 2:
                                        st.markdown(
                                            "<hr style='margin: 5px 0; border-color: rgba(255,255,255,0.1);'>",
                                            unsafe_allow_html=True,
                                        )
                            else:
                                st.info(
                                    f"No specific courses found for '{skill_title}' in the initial scan."
                                )
                        else:
                            st.warning(
                                "Course recommendations unavailable (ontology error)."
                            )
                        st.markdown(
                            f"[Find more courses for {skill_title} in the 'All Recommended Courses' tab](#)",
                            help="Click the 'All Recommended Courses' tab.",
                        )

        with tab2:
            st.markdown("Explore roles similar to the one you analysed:")
            st.markdown('<div class="results-section-card">', unsafe_allow_html=True)
            st.markdown("### 💼 Similar Job Roles")
            current_job_title = st.session_state.get("job_title")
            if not current_job_title:
                st.info("Job title not provided. Cannot search for similar roles.")
            else:
                esco_cache_key = (
                    f"esco_data_{current_job_title.lower().replace(' ', '_')}"
                )
                if esco_cache_key not in st.session_state:
                    with st.spinner(
                        f"Finding similar job roles for '{current_job_title}'…"
                    ):
                        st.session_state[esco_cache_key] = search_esco_occupations_api(
                            current_job_title
                        )
                occ_data = st.session_state.get(esco_cache_key, {})
                if occ_data and occ_data.get("similarOccupations"):
                    grouped = {}
                    for occ in occ_data["similarOccupations"]:
                        if occ.get("title"):
                            grouped.setdefault(
                                occ.get("relation", "Similar"), []
                            ).append(occ["title"])
                    if grouped:
                        for relation, titles in grouped.items():
                            with st.expander(f"{relation} ({len(titles)})"):
                                for t in titles:
                                    st.markdown(f"- {t}")
                    else:
                        st.info("No similar roles found via ESCO API.")
                else:
                    st.info("No similar roles found via ESCO API.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            if not missing:
                st.info("No course recommendations needed as no skill gaps were found.")
            else:
                st.markdown(
                    "Find courses to help you bridge the skill gap. Use the filters and click 'Get Recommendations'."
                )
                st.markdown(
                    '<div class="results-section-card">', unsafe_allow_html=True
                )
                st.markdown("### 🔍 Filter & Generate Course List")
                col_rt, col_diff = st.columns(2)
                with col_rt:
                    res_type = st.selectbox(
                        "Resource Type",
                        [
                            "Any",
                            "Course",
                            "MOOC",
                            "Tutorial",
                            "Workshop",
                            "Video",
                            "Interactive",
                        ],
                        key="all_course_filter_type",
                    )
                with col_diff:
                    diff_level = st.selectbox(
                        "Difficulty Level",
                        ["Any", "Beginner", "Intermediate", "Advanced", "Expert"],
                        key="all_course_filter_difficulty",
                    )
                res_type_filter = None if res_type == "Any" else res_type
                diff_level_filter = None if diff_level == "Any" else diff_level

                if st.button(
                    "🚀 Get/Refresh Full Course Recommendations",
                    key="get_all_courses_btn",
                ):
                    st.session_state.recommendations_generated = False
                    with st.spinner(
                        "Finding and processing all course recommendations..."
                    ):
                        try:
                            if (
                                st.session_state.ontology_graph is None
                                or st.session_state.ontology_skills_map is None
                            ):
                                st.error(
                                    "Ontology data not loaded. Cannot generate recommendations."
                                )
                                st.stop()

                            g_courses_all = st.session_state.ontology_graph
                            map_courses_all = st.session_state.ontology_skills_map

                            recs_all, _ = recommend_learning_resources(
                                missing,
                                g_courses_all,
                                map_courses_all,
                                st.session_state.max_courses,
                            )

                            def filter_recs(recs, r_type=None, diff=None):
                                out = {}
                                for sk, cs in recs.items():
                                    selected = [
                                        c
                                        for c in cs
                                        if (
                                            not r_type
                                            or c.get("resource_type", "").lower()
                                            == r_type.lower()
                                        )
                                        and (
                                            not diff
                                            or c.get("difficulty", "").lower()
                                            == diff.lower()
                                        )
                                    ]
                                    if selected:
                                        out[sk] = selected
                                return out

                            filtered_recs = filter_recs(
                                recs_all, res_type_filter, diff_level_filter
                            )

                            unique_courses = {}
                            for sk, cs in filtered_recs.items():
                                for c in cs:
                                    key = f"{c['course']}:{c['provider']}"
                                    if key not in unique_courses:
                                        c_copy = {**c, "skills": [sk]}
                                        unique_courses[key] = c_copy
                                    elif sk not in unique_courses[key]["skills"]:
                                        unique_courses[key]["skills"].append(sk)

                            processed_results_list = []
                            if unique_courses:
                                courses_to_fetch = list(unique_courses.values())
                                with concurrent.futures.ThreadPoolExecutor(
                                    max_workers=5
                                ) as ex:
                                    futures = {}
                                    for d in courses_to_fetch:
                                        try:
                                            future = ex.submit(
                                                process_course_for_link, d
                                            )
                                            futures[future] = d
                                        except Exception as exc:
                                            print(
                                                f"Error submitting task for {d.get('course')}: {exc}"
                                            )

                                    prog_bar = st.progress(
                                        0, text="Fetching course links..."
                                    )
                                    completed_count = 0
                                    total_futures = len(futures)

                                    for fut in concurrent.futures.as_completed(futures):
                                        completed_count += 1
                                        try:
                                            res = fut.result()
                                            if "link" not in res or not res["link"]:
                                                res["link"] = None
                                            processed_results_list.append(res)
                                        except Exception as exc:
                                            course_data = futures[fut]
                                            print(
                                                f"Error fetching link for {course_data.get('course')}: {exc}"
                                            )
                                            course_data["link"] = None
                                            course_data["search_method"] = "Error"
                                            processed_results_list.append(course_data)

                                        if total_futures > 0:
                                            prog_bar.progress(
                                                completed_count / total_futures,
                                                text=f"Fetching link {completed_count}/{total_futures}...",
                                            )

                                    time.sleep(0.5)
                                    prog_bar.empty()

                            processed_results_list.sort(
                                key=lambda x: x.get("relevance_score", 0), reverse=True
                            )

                            st.session_state.processed_recommendations = (
                                processed_results_list
                            )
                            st.session_state.recommendations_generated = True
                            st.success(
                                f"Found {len(processed_results_list)} unique course recommendations!"
                            )

                        except Exception as e:
                            st.error(f"Error generating course recommendations: {e}")
                            traceback.print_exc()
                            st.session_state.processed_recommendations = []
                            st.session_state.recommendations_generated = False

                st.markdown("</div>", unsafe_allow_html=True)

                if st.session_state.get("recommendations_generated", False):
                    processed_results_display = st.session_state.get(
                        "processed_recommendations", []
                    )
                    st.markdown("---")
                    st.markdown(
                        f"### Recommended Courses ({len(processed_results_display)} found)"
                    )

                    if not processed_results_display:
                        st.info("No courses found matching your criteria.")
                    else:
                        for c in processed_results_display:
                            with st.container(border=True):
                                st.markdown(f"#### {c['course']}")
                                skill_labels_display = []
                                requested_skills_sanitised = c.get("skills", [])
                                for skill_sanitised in requested_skills_sanitised:
                                    original_label = skill_sanitised
                                    if "ontology_skills_labels" in st.session_state:
                                        for (
                                            label
                                        ) in st.session_state.ontology_skills_labels:
                                            if (
                                                sanitise_skill(label).lower()
                                                == skill_sanitised.lower()
                                            ):
                                                original_label = label
                                                break
                                    skill_labels_display.append(
                                        original_label.replace("_", " ").title()
                                    )

                                st.markdown(
                                    f"**Recommended for:** {', '.join(skill_labels_display)}"
                                )

                                if c.get(
                                    "is_related_recommendation"
                                ):  # Check if it's broader/narrower
                                    actual_skill = c.get(
                                        "actual_skill_taught", "related skill"
                                    )
                                    st.markdown(
                                        f"<small><i>ℹ️ Recommended via related skill: **{actual_skill}**</i></small>",
                                        unsafe_allow_html=True,
                                    )

                                meta_col, link_col = st.columns([3, 1])

                                with meta_col:
                                    meta = []
                                    if c.get("provider") != "Unknown":
                                        meta.append(f"🎓 Provider: {c['provider']}")
                                    if c.get("difficulty"):
                                        meta.append(
                                            f"📊 Level: {get_difficulty_badge(c['difficulty'])}"
                                        )
                                    else:
                                        meta.append("📊 Level: Not Specified")
                                    if c.get("duration"):
                                        meta.append(f"⏱️ Duration: {c['duration']} h")
                                    if c.get("resource_type"):
                                        meta.append(f"📘 Type: {c['resource_type']}")
                                    st.markdown(
                                        " | ".join(meta), unsafe_allow_html=True
                                    )

                                with link_col:
                                    if c.get("link"):
                                        st.link_button(
                                            "View Course 🔗", c["link"], type="primary"
                                        )
                                    else:
                                        st.markdown("🔗 *Link unavailable*")

                                    button_key = f"add_{c.get('uri', c['course'])}_{c['provider']}".replace(
                                        " ", "_"
                                    ).replace(":", "_")
                                    if st.button("➕ Add to Journey", key=button_key):
                                        journey_item = {
                                            "name": c["course"],
                                            "skill": ", ".join(skill_labels_display),
                                            "status": "planned",
                                            "completion": 0,
                                            "added_date": time.strftime("%Y-%m-%d"),
                                            "link": c.get("link"),
                                        }
                                        if "learning_progress" not in st.session_state:
                                            st.session_state.learning_progress = {
                                                "planned": [],
                                                "in_progress": [],
                                                "completed": [],
                                            }
                                        if (
                                            "planned"
                                            not in st.session_state.learning_progress
                                        ):
                                            st.session_state.learning_progress[
                                                "planned"
                                            ] = []

                                        if not any(
                                            item["name"] == journey_item["name"]
                                            and item.get("provider") == c["provider"]
                                            for item in st.session_state.learning_progress[
                                                "planned"
                                            ]
                                        ):
                                            st.session_state.learning_progress[
                                                "planned"
                                            ].append(journey_item)
                                            st.success(
                                                f"Added '{c['course']}' to your planned journey!"
                                            )
                                        else:
                                            st.info(
                                                f"'{c['course']}' is already in your planned journey."
                                            )

                elif "recommendations_generated" in st.session_state:
                    if not st.session_state.get("processed_recommendations"):
                        st.warning(
                            "No recommendations generated yet or none found. Click the button above."
                        )


def skills_page():
    st.header("Skills Database")
    st.markdown("---")
    g = Graph()
    g.parse("skill_ontology.ttl", format="turtle")
    ONTO_URI = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology#"
    ONTO = Namespace(ONTO_URI)
    all_skills_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX onto: <{ONTO_URI}>
        SELECT ?s ?label WHERE {{
            ?s rdf:type onto:Skill ;
               rdfs:label ?label .
        }}
        ORDER BY ?label
    """
    skills_label_uri_list = [
        (str(r.label), str(r.s)) for r in g.query(all_skills_query)
    ]
    st.write("Search for a skill to see details:")
    query_params = st.query_params
    param_skill_sanitised = query_params.get("skill", [None])[0]
    current_query_in_box = st.session_state.get("skill_query_input", "")
    initial_query = current_query_in_box
    if param_skill_sanitised and "skill_param_processed" not in st.session_state:
        found_label = None
        for lbl, _ in skills_label_uri_list:
            if sanitise_skill(lbl) == param_skill_sanitised:
                found_label = lbl
                break
        if found_label:
            initial_query = found_label
            st.session_state.skill_param_processed = True
            st.session_state.skill_query_input = found_label
            st.rerun()
        else:
            st.query_params.clear()
            st.session_state.skill_param_processed = True
            initial_query = ""
            st.session_state.skill_query_input = ""
    query = st.text_input(
        "Enter skill name", value=initial_query, key="skill_query_input"
    )
    if "skill_param_processed" in st.session_state and query != initial_query:
        del st.session_state.skill_param_processed
        st.query_params.clear()
    if not query:
        st.info("Enter a skill name above to search.")
        return
    matches = [s for s in skills_label_uri_list if query.lower() == s[0].lower()]
    if not matches:
        matches = [s for s in skills_label_uri_list if query.lower() in s[0].lower()]
    if not matches:
        st.warning("No skills matched your search.")
        return
    # Only show the ontology skill selectbox if there are multiple ontology matches
    if len(matches) == 1:
        skill_label, skill_uri = matches[0]
    else:
        labels = [l for l, _ in matches]
        choice = st.selectbox(
            f"Found {len(matches)} skills matching '{query}'. Please choose one:",
            labels,
            key=f"ontology_choice_{query}",
        )
        skill_label = choice
        skill_uri = next(u for l, u in matches if l == choice)
    # Now get ESCO matches for the selected ontology skill
    hits = list_esco_matches(skill_label)
    if not hits:
        esco_data = None
    elif len(hits) == 1:
        selected_title = hits[0]["title"]
        esco_data = search_esco_api(selected_title)
    else:
        # Only show the ESCO selectbox if there are multiple ESCO matches
        titles = [h["title"] for h in hits]
        label_map = {capwords(h["title"]): h for h in hits}
        choice = st.selectbox(
            f"Found {len(hits)} results for '{skill_label}'. Which did you mean?",
            list(label_map.keys()),
            key=f"esco_choice_{skill_label}",
        )
        selected_title = label_map[choice]["title"]
        esco_data = search_esco_api(selected_title)

    if esco_data:
        st.subheader(f"Details for: {esco_data['title']}")
        tabs = st.tabs(
            ["Description", "Skill Relationships", "Related Occupations", "Courses"]
        )
        desc_tab, rel_tab, occ_tab, course_tab = tabs
        with desc_tab:
            st.markdown("### Description")
            st.write(esco_data.get("description", "No description available."))
        with rel_tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Broader Skills ")
                # Get the list, default to empty list if key doesn't exist
                broader = esco_data.get("broaderSkills", [])
                # Check if the list is not empty BEFORE looping
                if broader:
                    for s in broader:
                        # Ensure s is a string before writing
                        st.write(f"- {str(capwords(s))}")
                else:
                    # This message shows only if the list is empty
                    st.write("No broader skills found.")
            with col2:
                st.markdown("### Narrower Skills")
                # Get the list, default to empty list if key doesn't exist
                narrower = esco_data.get("narrowerSkills", [])
                # Check if the list is not empty BEFORE looping
                if narrower:
                    for s in narrower:
                        # Ensure s is a string before writing
                        st.write(f"- {str(capwords(s))}")
                else:
                    # This message shows only if the list is empty
                    st.write("No narrower skills found.")

            st.markdown("### Some Related Occupations")
            occs = esco_data.get("occupations", [])
            if occs:
                essential = [o for o in occs if "(Essential)" in o]
                optional = [o for o in occs if "(Optional)" in o]
                st.metric("Total Occupations", len(occs))
                c1, c2 = st.columns(2)
                c1.metric("Essential For", len(essential))
                c2.metric("Optional For", len(optional))
                if essential:
                    with st.expander("Essential For", expanded=True):
                        for o in essential:
                            st.write(f"- {o.replace(' (Essential)', '').strip()}")
                if optional:
                    with st.expander("Optional For", expanded=True):
                        for o in optional:
                            st.write(f"- {o.replace(' (Optional)', '').strip()}")
            else:
                st.write("No related occupations found.")
        course_container = course_tab
    else:
        st.subheader(f"Details for: {skill_label}")
        st.info(
            f"No additional information found for '{skill_label}' in the ESCO database."
        )
        st.markdown("---")
        st.subheader("Available Courses")
        course_container = st.container()
    sparql_courses = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX onto: <{ONTO_URI}>
    SELECT DISTINCT ?res ?label ?providerName ?rating ?duration ?level ?commitment ?resourceType ?taughtSkillLabel ?isDirectMatch
    WHERE {{
        {{
            BIND(<{skill_uri}> AS ?targetSkill)
            BIND(true AS ?isDirectMatch)
        }}
        UNION
        {{
            {{ <{skill_uri}> onto:relatedSkill ?targetSkill . }}
            UNION
            {{ ?targetSkill onto:relatedSkill <{skill_uri}> . }}
            FILTER(?targetSkill != <{skill_uri}>)
            BIND(false AS ?isDirectMatch)
        }}
        ?res rdf:type onto:Learning_Resource ;
             rdfs:label ?label ;
             onto:skillTaught ?targetSkill .
        ?targetSkill rdfs:label ?taughtSkillLabel .
        OPTIONAL {{ ?res onto:providedBy ?prov . ?prov rdfs:label ?providerName. }}
        OPTIONAL {{ ?res onto:hasRating ?rating. }}
        OPTIONAL {{ ?res onto:estimatedDuration ?duration. }}
        OPTIONAL {{ ?res onto:hasDifficultyLevel ?levelUri . ?levelUri rdfs:label ?level . }}
        OPTIONAL {{ ?res onto:timeCommitmentCategory ?commitment. }}
        OPTIONAL {{ ?res onto:hasResourceType ?typeUri . ?typeUri rdfs:label ?resourceType . }}
        OPTIONAL {{ ?res rdf:type onto:HighlyRatedResource . BIND(true AS ?isHR) }}
    }}
    ORDER BY ?label
    """
    raw_results = list(g.query(sparql_courses))
    with course_container:
        if not raw_results:
            st.info(
                f"No courses found in the ontology that teach '{skill_label}' or related skills."
            )
            return
        courses = []
        for r in raw_results:
            is_highly_rated = (
                bool(r.isHR)
                if hasattr(r, "isHR") and r.isHR
                else (r.res, RDF.type, ONTO.HighlyRatedResource) in g
            )
            taught_skill = (
                str(r.taughtSkillLabel) if hasattr(r, "taughtSkillLabel") else None
            )
            is_direct = bool(r.isDirectMatch) if hasattr(r, "isDirectMatch") else False
            c = {
                "uri": str(r.res),
                "label": str(r.label),
                "provider": (
                    str(r.providerName)
                    if hasattr(r, "providerName") and r.providerName
                    else "—"
                ),
                "is_highly_rated": is_highly_rated,
                "rating": None,
                "duration": None,
                "level": None,
                "commitment": None,
                "resourceType": None,
                "taught_skill": taught_skill,
                "is_direct_match": is_direct,
            }
            if hasattr(r, "rating") and r.rating:
                try:
                    c["rating"] = float(r.rating)
                except:
                    pass
            if hasattr(r, "duration") and r.duration:
                try:
                    c["duration"] = float(r.duration)
                except:
                    pass
            if hasattr(r, "level") and r.level:
                l = str(r.level).lower()
                c["level"] = (
                    "Beginner"
                    if "beginner" in l
                    else (
                        "Intermediate"
                        if "intermediate" in l
                        else (
                            "Advanced"
                            if "advanced" in l
                            else "Expert"
                            if "expert" in l
                            else str(r.level)
                        )
                    )
                )
            if hasattr(r, "commitment") and r.commitment:
                c["commitment"] = str(r.commitment).replace('"', "")
            if hasattr(r, "resourceType") and r.resourceType:
                rt = str(r.resourceType)
                c["resourceType"] = rt.split("#")[-1] if "#" in rt else rt
            courses.append(c)
        sort_option = st.selectbox(
            "Sort courses by:",
            [
                "Relevance (Direct Skill First)",
                "Rating: High → Low",
                "Rating: Low → High",
                "Course Name (A → Z)",
                "Provider (A → Z)",
                "Duration: Short → Long",
                "Duration: Long → Short",
            ],
            key=f"sort_{skill_uri}",
        )
        if sort_option == "Relevance (Direct Skill First)":
            courses.sort(
                key=lambda x: (
                    x["is_direct_match"],
                    x["rating"] if x["rating"] is not None else -1,
                ),
                reverse=True,
            )
        elif sort_option == "Rating: High → Low":
            courses.sort(
                key=lambda x: x["rating"] if x["rating"] is not None else -1,
                reverse=True,
            )
        elif sort_option == "Rating: Low → High":
            courses.sort(
                key=lambda x: x["rating"] if x["rating"] is not None else float("inf")
            )
        elif sort_option == "Course Name (A → Z)":
            courses.sort(key=lambda x: x["label"].lower())
        elif sort_option == "Provider (A → Z)":
            courses.sort(key=lambda x: x["provider"].lower())
        elif sort_option == "Duration: Short → Long":
            courses.sort(
                key=lambda x: (
                    x["duration"] if x["duration"] is not None else float("inf")
                )
            )
        elif sort_option == "Duration: Long → Short":
            courses.sort(
                key=lambda x: x["duration"] if x["duration"] is not None else -1,
                reverse=True,
            )
        st.write(
            f"Found {len(courses)} courses for '{skill_label}' and related skills."
        )
        for c in courses:
            with st.expander(
                f"{c['label']} {'(Direct Match)' if c['is_direct_match'] else ''}"
            ):
                with st.container(border=True):
                    if not c["is_direct_match"] and c["taught_skill"]:
                        st.markdown(
                            f"<small><i>Relevant for related skill: {c['taught_skill']}</i></small>",
                            unsafe_allow_html=True,
                        )
                    elif (
                        c["taught_skill"]
                        and c["taught_skill"].lower() != skill_label.lower()
                    ):
                        st.markdown(
                            f"<small><i>Teaches skill: {c['taught_skill']}</i></small>",
                            unsafe_allow_html=True,
                        )
                    badges = (
                        '<div class="badge-container" style="margin-bottom: 15px;">'
                    )
                    if c["is_highly_rated"]:
                        badges += '<span class="badge highly-rated-badge">⭐ HIGHLY RATED</span>'
                    if c["level"]:
                        badges += f'<span class="badge {c["level"].lower()}-badge">{c["level"].upper()}</span>'
                    badges += "</div>"
                    st.markdown(badges, unsafe_allow_html=True)
                    st.markdown(f"**Provider:** {c['provider']}")
                    st.markdown("---")
                    mc = st.columns([2, 2, 2, 1])
                    with mc[0]:
                        if c["duration"]:
                            d = c["duration"]
                            st.markdown(
                                f"⏱️ **Duration**<br>{int(d) if d == int(d) else d} hours",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown("⏱️ **Duration**<br>N/A", unsafe_allow_html=True)
                    with mc[1]:
                        st.markdown(
                            f"📘 **Type**<br>{c['resourceType'] if c['resourceType'] else 'N/A'}",
                            unsafe_allow_html=True,
                        )
                    with mc[2]:
                        st.markdown(
                            f"⏳ **Commitment**<br>{c['commitment'] if c['commitment'] else 'N/A'}",
                            unsafe_allow_html=True,
                        )
                    with mc[3]:
                        if c["rating"] is not None:
                            st.markdown(
                                f"⭐ **Rating**<br><span style='font-size:1.2em;font-weight:bold;'>{c['rating']:.1f}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown("⭐ **Rating**<br>N/A", unsafe_allow_html=True)


def courses_page():
    st.markdown(
        "<h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>Course Explorer</h1>",
        unsafe_allow_html=True,
    )

    # Add custom CSS for better styling
    st.markdown(
        """
    <style>
        .course-card {
            background-color: rgba(49, 51, 63, 0.7);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #4285F4;
            transition: transform 0.2s;
        }
        .course-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .course-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: white;
        }
        .provider-badge {
            background-color: #34A853;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 10px;
        }
        .rating-stars {
            color: gold;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        .skill-tag {
            background-color: #4285F4;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            display: inline-block;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .course-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            color: #BBBBBB;
            font-size: 0.9rem;
        }
        .filter-section {
            background-color: rgba(49, 51, 63, 0.4);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .pagination {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .page-info {
            margin: 0 15px;
            padding-top: 5px;
            color: #BBBBBB;
        }
        .page-btn {
            background-color: #4285F4;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        .page-btn:disabled {
            background-color: #666;
            cursor: not-allowed;
        }
        .view-course-btn {
            background-color: #4CAF50;
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9rem;
            transition: background-color 0.3s;
            display: inline-block;
            margin-right: 10px;
        }
        .view-course-btn:hover {
            background-color: #45a049;
        }
        .add-journey-btn {
            color: #4285F4;
            text-decoration: none;
            font-size: 0.9rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Define the getLinks function if it's not already defined elsewhere
    def getLinks(course_title, provider, show_debug_in_main_thread=False):
        """
        Function to get course links using search engine API
        Returns: (link_url, search_method)
        """
        # Import the necessary modules for SERP API
        try:
            import json
            import os
            from serpapi import GoogleSearch
            from dotenv import load_dotenv
        except ImportError:
            st.error(
                "Required modules not installed. Please install: pip install google-search-results python-dotenv"
            )
            return "#", "error"

        # Load the API key from .env file
        env_path = os.path.join(
            os.path.expanduser("~"), "Documents", "HONOURS-WORK", ".env"
        )
        load_dotenv(env_path)

        # Get API key from environment variable
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            if show_debug_in_main_thread:
                st.warning("SERPAPI_API_KEY not found in .env file")
            return "#", "no_api_key"

        try:
            # search query
            search_query = f"{course_title} {provider} course"

            # search parameters
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": api_key,
                "num": 5,  # Limit to top 5 results
            }

            # Execute search
            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract top organic result
            if "organic_results" in results and len(results["organic_results"]) > 0:
                top_result = results["organic_results"][0]
                return top_result.get("link", "#"), "serp_api"
            else:
                return "#", "no_results"

        except Exception as e:
            if show_debug_in_main_thread:
                st.error(f"Error searching for course: {e}")
            return "#", f"error: {str(e)}"

    # Load ontology graph and namespaces
    g, onto, data, ontology_skills_labels, ontology_skills_map = load_ontology()
    onto = "http://www.semanticweb.org/ukanw/ontologies/2025/2/skill-ontology"

    # Filter and search section
    st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Provider filter
        providers_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX onto: <{onto}#>
        SELECT DISTINCT ?providerName WHERE {{
            ?res onto:providedBy ?prov .
            ?prov rdfs:label ?providerName .
        }}
        ORDER BY ?providerName
        """
        try:
            providers_results = list(g.query(providers_query))
            providers = ["All Providers"] + [
                str(row.providerName) for row in providers_results
            ]
            selected_provider = st.selectbox("Provider", providers)
        except Exception as e:
            st.error(f"Error fetching providers: {e}")
            selected_provider = "All Providers"

    with col2:
        # Rating filter
        ratings = ["All Ratings", "4.9", "4.8+", "4.5+", "4.0+"]
        selected_rating = st.selectbox("Minimum Rating", ratings)

    with col3:
        # Text search
        search_term = st.text_input("Search Courses", "")

    # Difficulty filter
    difficulties = ["All Levels", "Beginner", "Intermediate", "Advanced", "Expert"]
    selected_difficulty = st.selectbox("Difficulty Level", difficulties)

    st.markdown("</div>", unsafe_allow_html=True)

    # Build the query
    base_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX onto: <{onto}#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?res ?label ?providerName ?rating ?description ?difficulty ?duration ?skillLabel WHERE {{
        ?res rdf:type onto:Learning_Resource .
        ?res rdfs:label ?label .
        OPTIONAL {{ ?res onto:hasRating ?rating . }}
        OPTIONAL {{ ?res onto:providedBy ?prov . ?prov rdfs:label ?providerName . }}
        OPTIONAL {{ ?res onto:description ?description . }}
        OPTIONAL {{ ?res onto:hasDifficultyLevel ?diffLevel . ?diffLevel rdfs:label ?difficulty . }}
        OPTIONAL {{ ?res onto:estimatedDuration ?duration . }}
        OPTIONAL {{ ?res onto:skillTaught ?skill . ?skill rdfs:label ?skillLabel . }}
    """

    # Add filters
    if selected_provider != "All Providers":
        base_query += f'FILTER(?providerName = "{selected_provider}") .\n        '

    if selected_rating != "All Ratings":
        if selected_rating == "4.9":
            base_query += 'FILTER(?rating = "4.9") .\n        '
        elif selected_rating == "4.8+":
            base_query += "FILTER(xsd:float(?rating) >= 4.8) .\n        "
        elif selected_rating == "4.5+":
            base_query += "FILTER(xsd:float(?rating) >= 4.5) .\n        "
        elif selected_rating == "4.0+":
            base_query += "FILTER(xsd:float(?rating) >= 4.0) .\n        "

    if selected_difficulty != "All Levels":
        difficulty_lower = selected_difficulty.lower()
        base_query += f'FILTER(CONTAINS(LCASE(STR(?difficulty)), "{difficulty_lower}")) .\n        '

    # Close the query
    base_query += "} ORDER BY ?label"

    try:
        results = list(g.query(base_query))
    except Exception as e:
        st.error(f"Error querying ontology for courses: {e}")
        st.code(base_query, language="sparql")
        return

    # Process results
    courses_by_id = {}

    for res in results:
        course_id = str(res.res)

        if course_id not in courses_by_id:
            courses_by_id[course_id] = {
                "Title": str(res.label),
                "Provider": (
                    str(res.providerName)
                    if hasattr(res, "providerName") and res.providerName
                    else "Unknown"
                ),
                "Rating": (
                    float(res.rating) if hasattr(res, "rating") and res.rating else None
                ),
                "Description": (
                    str(res.description)
                    if hasattr(res, "description") and res.description
                    else None
                ),
                "Difficulty": (
                    str(res.difficulty)
                    if hasattr(res, "difficulty") and res.difficulty
                    else None
                ),
                "Duration": (
                    float(res.duration)
                    if hasattr(res, "duration") and res.duration
                    else None
                ),
                "Skills": [],
            }

        # Add skill if present
        if hasattr(res, "skillLabel") and res.skillLabel:
            skill = str(res.skillLabel)
            if skill not in courses_by_id[course_id]["Skills"]:
                courses_by_id[course_id]["Skills"].append(skill)

    # Filter by search term if provided
    if search_term:
        search_term_lower = search_term.lower()
        courses_by_id = {
            k: v
            for k, v in courses_by_id.items()
            if search_term_lower in v["Title"].lower()
            or any(search_term_lower in skill.lower() for skill in v["Skills"])
        }

    # Get the list of courses for pagination
    courses_list = list(courses_by_id.values())

    # Add course statistics at the TOP after filters
    st.markdown("---")
    st.subheader("Course Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        rated_courses = [c for c in courses_list if c["Rating"]]
        avg_rating = (
            sum(c["Rating"] for c in rated_courses) / len(rated_courses)
            if rated_courses
            else 0
        )
        st.metric("Average Rating", f"{avg_rating:.1f}/5.0")

    with col2:
        providers = set(
            c["Provider"] for c in courses_list if c["Provider"] != "Unknown"
        )
        st.metric("Total Providers", len(providers))

    with col3:
        all_skills = set()
        for course in courses_list:
            all_skills.update(course["Skills"])
        st.metric("Skills Covered", len(all_skills))

    st.markdown("---")

    # Set up pagination
    courses_per_page = 10
    total_pages = max(1, (len(courses_list) + courses_per_page - 1) // courses_per_page)

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Functions to handle page navigation
    def next_page():
        st.session_state.current_page = min(
            st.session_state.current_page + 1, total_pages
        )

    def prev_page():
        st.session_state.current_page = max(st.session_state.current_page - 1, 1)

    # Reset page when filters change
    filter_key = (
        f"{selected_provider}_{selected_rating}_{selected_difficulty}_{search_term}"
    )
    if (
        "last_filter" not in st.session_state
        or st.session_state.last_filter != filter_key
    ):
        st.session_state.current_page = 1
        st.session_state.last_filter = filter_key

    # Display course count and pagination info
    st.markdown(f"<h2>Found {len(courses_list)} Courses</h2>", unsafe_allow_html=True)

    if not courses_list:
        st.info("No courses match your criteria. Try adjusting your filters.")
        return

    # Calculate the start and end indices for current page
    start_idx = (st.session_state.current_page - 1) * courses_per_page
    end_idx = min(start_idx + courses_per_page, len(courses_list))

    # Display pagination controls (TOP)
    col1_top, col2_top, col3_top = st.columns([1, 2, 1])
    with col1_top:
        if st.session_state.current_page > 1:
            st.button("← Previous Page", on_click=prev_page, key="prev_top")

    with col2_top:
        st.markdown(
            f"<div class='page-info'>Page {st.session_state.current_page} of {total_pages}</div>",
            unsafe_allow_html=True,
        )

    with col3_top:
        if st.session_state.current_page < total_pages:
            st.button("Next Page →", on_click=next_page, key="next_top")

    # Display only the courses for the current page
    for course in courses_list[start_idx:end_idx]:
        # Generate star rating
        stars = "★" * int(course["Rating"] if course["Rating"] else 0)
        if course["Rating"] and course["Rating"] % 1 >= 0.5:
            stars += "½"

        # Format difficulty badge
        difficulty_html = ""
        if course["Difficulty"]:
            difficulty_color = (
                "#4CAF50"
                if "beginner" in course["Difficulty"].lower()
                else (
                    "#FFC107"
                    if "intermediate" in course["Difficulty"].lower()
                    else (
                        "#F44336"
                        if "advanced" in course["Difficulty"].lower()
                        else (
                            "#9C27B0"
                            if "expert" in course["Difficulty"].lower()
                            else "#607D8B"
                        )
                    )
                )
            )
            difficulty_html = f"""<span style="background-color:{difficulty_color};color:white;padding:3px 8px;border-radius:12px;font-size:0.8rem;margin-left:10px;">{course["Difficulty"]}</span>"""

        # Format skills tags
        skills_html = ""
        for skill in course["Skills"][:3]:  # Limit to first 3 skills
            skills_html += f'<span class="skill-tag">{skill}</span>'
        if len(course["Skills"]) > 3:
            skills_html += (
                f'<span class="skill-tag">+{len(course["Skills"]) - 3} more</span>'
            )

        # Generate course link with SERP API
        course_link, search_method = getLinks(
            course["Title"], course["Provider"], show_debug_in_main_thread=False
        )
        course_link = course_link if course_link else "#"

        st.markdown(
            f"""
        <div class="course-card">
            <div class="course-title">{course["Title"]}</div>
            <div class="provider-badge">{course["Provider"]}{difficulty_html}</div>
            <div class="rating-stars">{stars} <span style="color:white;font-size:0.9rem;">({course["Rating"] if course["Rating"] else "No rating"})</span></div>
            <div>{skills_html}</div>
            <div class="course-meta">
                <div>{f"Duration: {course['Duration']} hours" if course["Duration"] else ""}</div>
                <div>
                    <a href="{course_link}" target="_blank" class="view-course-btn">View Course</a>
                    <a href="#" class="add-journey-btn">Add to Journey</a>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Display pagination controls at bottom
    st.markdown("<div class='pagination'>", unsafe_allow_html=True)
    col1_bottom, col2_bottom, col3_bottom = st.columns([1, 2, 1])
    with col1_bottom:
        if st.session_state.current_page > 1:
            st.button("← Previous", on_click=prev_page, key="prev_bottom")

    with col2_bottom:
        st.markdown(
            f"<div class='page-info'>Page {st.session_state.current_page} of {total_pages} ({start_idx + 1}-{end_idx} of {len(courses_list)})</div>",
            unsafe_allow_html=True,
        )

    with col3_bottom:
        if st.session_state.current_page < total_pages:
            st.button("Next →", on_click=next_page, key="next_bottom")

    st.markdown("</div>", unsafe_allow_html=True)


def progress_page():
    st.title("My Learning Journey")

    # Add new resource to progress
    with st.expander("Add Resource to My Journey"):
        col1, col2 = st.columns(2)
        with col1:
            resource_name = st.text_input("Resource Name")
            skill_name = st.text_input("Related Skill")
        with col2:
            status = st.selectbox("Status", ["Planned", "In Progress", "Completed"])
            completion = st.slider("Completion %", 0, 100, 0)

        if st.button("Add to My Journey") and resource_name:
            new_item = {
                "name": resource_name,
                "skill": skill_name,
                "status": status.lower().replace(" ", "_"),
                "completion": completion,
                "added_date": time.strftime("%Y-%m-%d"),
            }

            status_key = (
                f"{status.lower().replace(' ', '_')}s"
                if status != "In Progress"
                else "in_progress"
            )
            if status_key not in st.session_state.learning_progress:
                st.session_state.learning_progress[status_key] = []

            st.session_state.learning_progress[status_key].append(new_item)
            st.success(f"Added {resource_name} to your learning journey!")

    # Dashboard stats
    col1, col2, col3 = st.columns(3)

    with col1:
        completed_count = len(st.session_state.learning_progress.get("completed", []))
        st.metric("Completed Resources", completed_count)

    with col2:
        in_progress_count = len(
            st.session_state.learning_progress.get("in_progress", [])
        )
        st.metric("Resources In Progress", in_progress_count)

    with col3:
        planned_count = len(st.session_state.learning_progress.get("planned", []))
        st.metric("Planned Resources", planned_count)

    # Timeline of learning journey
    st.subheader("My Learning Timeline")

    all_items = []
    for status, items in st.session_state.learning_progress.items():
        for item in items:
            item["status_name"] = status
            all_items.append(item)

    # Sort by added date
    all_items.sort(key=lambda x: x.get("added_date", ""))

    for item in all_items:
        status = item.get("status_name", "")
        completion = item.get("completion", 0)

        if "completed" in status:
            color = "#28a745"  # Green
            emoji = "✅"
        elif "in_progress" in status:
            color = "#ffc107"  # Yellow
            emoji = "🔄"
        else:
            color = "#6c757d"  # Gray
            emoji = "📝"

        st.markdown(
            f"""
        <div style="border-left: 3px solid {
                color
            }; padding-left: 15px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between;">
                <h4 style="margin: 0;">{emoji} {item["name"]}</h4>
                <span style="color: #6c757d;">{item.get("added_date", "")}</span>
            </div>
            <p style="margin: 5px 0 0 0;">Skill: {
                item.get("skill", "Not specified")
            }</p>
            {
                "<div style='width: 100%; background-color: #e9ecef; height: 10px; border-radius: 5px; margin-top: 8px;'>"
                f"<div style='width: {completion}%; background-color: {color}; height: 100%; border-radius: 5px;'></div></div>"
                if "in_progress" in status
                else ""
            }
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    load_css()
    create_navbar()

    # Use st.query_params to get the query parameters
    query_params = st.query_params
    # Access the 'page' key, defaulting to 'home' if not present
    current_page = query_params.get("page", "home").lower()

    if "learning_progress" not in st.session_state:
        st.session_state.learning_progress = {
            "completed_resources": [],
            "in_progress": [],
            "planned": [],
        }

    # Use st.query_params to get the query parameters
    query_params = st.query_params
    # Access the 'page' key, defaulting to 'home' if not present
    current_page = query_params.get("page", "home").lower()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if current_page == "home":
        home_page()
    elif current_page == "recommendations":
        recommendations_page()
    elif current_page == "skills":
        skills_page()
    elif current_page == "courses":
        courses_page()
    elif current_page == "progress":
        progress_page()
    else:
        st.error("Page not found!")
        home_page()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
    <style>
        .footer {
            position: relative;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #333;
            text-align: center;
            color: #888;
            font-size: 0.8rem;
            width: 100%;
        }
    </style>
    <div class="footer">
        Powered by Streamlit, RDFlib, SpaCy, and Hugging Face Transformers
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

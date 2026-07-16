import os
import json
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests

# Hardcoded base library
BASE_LIBRARY = [
    # Physics
    {"subdomain": "Physics_Kinematics", "misconception_name": "gravity_misconception", "symptom_in_diagram": "Student believes heavier objects fall faster than lighter ones regardless of air resistance.", "socratic_prompt_template": "If you drop a bowling ball and a golf ball from the same height in a vacuum, which hits the ground first?"},
    {"subdomain": "Physics_Kinematics", "misconception_name": "force_vector_error", "symptom_in_diagram": "Student confuses force and velocity, drawing force arrows parallel to motion instead of net resultant.", "socratic_prompt_template": "What is the difference between the direction an object is moving and the direction of the net force acting on it?"},
    {"subdomain": "Physics_Energy", "misconception_name": "energy_transfer_misconception", "symptom_in_diagram": "Student treats energy as a substance that is 'used up' rather than transferred or transformed.", "socratic_prompt_template": "Where does the energy 'go' when a battery dies?"},
    {"subdomain": "Physics_Electricity", "misconception_name": "current_consumption_error", "symptom_in_diagram": "Student believes current is consumed by a resistor rather than voltage drop occurring.", "socratic_prompt_template": "If current is 'used up' by the resistor, what would happen to the current leaving the resistor compared to entering it?"},
    {"subdomain": "Physics_Waves", "misconception_name": "wave_property_confusion", "symptom_in_diagram": "Student conflates frequency and amplitude, thinking louder sound has higher pitch.", "socratic_prompt_template": "How would you describe a sound that has high amplitude but low frequency?"},
    
    # Biology
    {"subdomain": "Biology_Cell_Biology", "misconception_name": "photosynthesis_misconception", "symptom_in_diagram": "Student believes plants get their food from the soil rather than producing it through photosynthesis.", "socratic_prompt_template": "If a plant gets its food entirely from the soil, what is the role of sunlight and leaves?"},
    {"subdomain": "Biology_Genetics", "misconception_name": "lamarckian_inheritance", "symptom_in_diagram": "Student believes organisms can pass on acquired characteristics to offspring.", "socratic_prompt_template": "If someone works out and builds large muscles, will their children naturally be born with larger muscles?"},
    {"subdomain": "Biology_Evolution", "misconception_name": "teleological_evolution", "symptom_in_diagram": "Student believes evolution has a goal or direction toward 'higher' organisms.", "socratic_prompt_template": "Does natural selection favor the 'most advanced' organism, or the one best adapted to its current environment?"},
    
    # Chemistry
    {"subdomain": "Chemistry_Atomic_Structure", "misconception_name": "bohr_model_overextension", "symptom_in_diagram": "Student draws electrons in fixed circular orbits around nucleus.", "socratic_prompt_template": "Are electrons restricted to fixed 2D planetary orbits, or do they exist in 3D probability clouds (orbitals)?"},
    {"subdomain": "Chemistry_Reactions", "misconception_name": "conservation_of_mass_error", "symptom_in_diagram": "Student believes a product 'disappears' in a reaction rather than mass being conserved.", "socratic_prompt_template": "If a piece of wood burns and leaves only a small pile of ash, where did the rest of the mass go?"},
]

def scrape_misconceptions():
    """Scrape additional educational misconceptions from a reliable source."""
    scraped = []
    try:
        url = "https://en.wikipedia.org/wiki/List_of_common_misconceptions"
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract list items
        for li in soup.find_all("li"):
            text = li.get_text()
            if len(text) > 50 and len(text) < 300:
                if "gravity" in text.lower() or "physics" in text.lower():
                    scraped.append({
                        "subdomain": "Physics_General",
                        "misconception_name": "general_physics_misc",
                        "symptom_in_diagram": text,
                        "socratic_prompt_template": "How does this scientific principle apply to the real world?"
                    })
                elif "biology" in text.lower() or "evolution" in text.lower():
                    scraped.append({
                        "subdomain": "Biology_General",
                        "misconception_name": "general_biology_misc",
                        "symptom_in_diagram": text,
                        "socratic_prompt_template": "What evidence supports this biological process?"
                    })
    except Exception as e:
        print(f"Scraping failed: {e}")
        
    return scraped

def build_taxonomy():
    print("Building taxonomy...")
    taxonomy = BASE_LIBRARY + scrape_misconceptions()
    
    # Save taxonomy JSON
    os.makedirs("data", exist_ok=True)
    with open("data/taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)
        
    print(f"Saved {len(taxonomy)} items to data/taxonomy.json")
    
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [item["symptom_in_diagram"] for item in taxonomy]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    faiss.write_index(index, "data/taxonomy.index")
    print("Saved FAISS index to data/taxonomy.index")

if __name__ == "__main__":
    build_taxonomy()

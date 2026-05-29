import re

def clean_stylometric_text(text):
    if not isinstance(text, str):
        return ""
        
    # 1. Strip Markdown Formatting (Bold, Italics, Headers, Lists)
    text = re.sub(r'\*\*|__', '', text)  # Removes bold markers
    text = re.sub(r'#+\s+', '', text)    # Removes header hashes
    text = re.sub(r'[`*_\-]', '', text)  # Removes remaining inline formatting tokens
    
    # 2. Strip Volatile Marketing Buzzwords (The specific corporate triggers)
    buzzwords = [
        "groundbreaking", "empowering", "revolutionize", "testament", 
        "delve", "foster", "tapestry", "subsequent", "pivotal", "seamlessly"
    ]
    for word in buzzwords:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
        
    # 3. Clean up any awkward double spaces left behind by deletions
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
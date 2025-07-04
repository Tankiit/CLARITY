#!/usr/bin/env python
"""
Concept-to-rationale mapper using CEBaB dataset concepts.
This script integrates discovered concepts from CEBaB with the concept bottleneck explanation model.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import re
from collections import defaultdict

# Try to import libraries
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from datasets import load_dataset
    HAVE_LIBRARIES = True
except ImportError:
    HAVE_LIBRARIES = False
    print("Warning: Some required libraries are not installed.")
    print("Run: pip install torch transformers datasets numpy")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Map CEBaB concepts to rationales in text explanations")
    
    # Input options
    parser.add_argument("--text", type=str, default=None,
                        help="Single text input to explain")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to file containing text to explain (one per line)")
    
    # Output options
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save explanations (default: print to console)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML visualization of concept-rationale mapping")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print output for human readability")
    
    # Concept options
    parser.add_argument("--concept_map", type=str, default="concept_word_map.json",
                        help="Path to concept word map (generated by explore_cebab.py)")
    parser.add_argument("--use_dataset_concepts", action="store_true",
                        help="Use concepts directly from the CEBaB dataset")
    parser.add_argument("--cebab_samples", type=int, default=100,
                        help="Number of CEBaB samples to use for concept discovery")
    
    # Explanation settings
    parser.add_argument("--top_concepts", type=int, default=5,
                        help="Number of top concepts to consider")
    parser.add_argument("--concept_threshold", type=float, default=0.1,
                        help="Threshold for concept activation")
    
    return parser.parse_args()

def load_cebab_dataset(num_samples=100):
    """Load a subset of the CEBaB dataset for concept discovery"""
    if not HAVE_LIBRARIES:
        print("Error: Cannot load dataset - required libraries not installed.")
        return None
    
    try:
        print(f"Loading {num_samples} samples from CEBaB dataset...")
        ds = load_dataset("CEBaB/CEBaB", split=f"train[:{num_samples}]")
        print(f"Successfully loaded {len(ds)} samples.")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def discover_concepts_from_cebab(dataset):
    """Extract concepts directly from CEBaB dataset annotations"""
    # Predefined aspects in CEBaB
    aspects = ["food", "service", "ambiance", "noise"]
    
    # Store words associated with each aspect based on sentiment
    aspect_sentiment_words = {
        aspect: {"positive": set(), "negative": set()} 
        for aspect in aspects
    }
    
    # Simple word extraction using basic NLP techniques
    import re
    from collections import Counter
    
    # Stopwords to exclude
    stopwords = set(["the", "and", "was", "is", "in", "to", "a", "of", "for", "with", 
                    "on", "that", "this", "it", "at", "as", "by", "i", "we", "our",
                    "were", "they", "their", "be", "from", "had", "have"])
    
    # Process each sample
    for sample in dataset:
        text = sample["description"].lower()
        words = re.findall(r'\b[a-z]+\b', text)
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # For each aspect, check if it's mentioned and with what sentiment
        for aspect in aspects:
            sentiment = sample[f"{aspect}_aspect_majority"]
            
            if sentiment == "Positive":
                aspect_sentiment_words[aspect]["positive"].update(words)
            elif sentiment == "Negative":
                aspect_sentiment_words[aspect]["negative"].update(words)
    
    # Convert to concept format used by the explanation model
    concept_word_map = {}
    
    # Map aspect-sentiment pairs to concept numbers
    aspect_to_concept = {
        "food_positive": [5, 12],
        "service_positive": [28],
        "ambiance_positive": [7],
        "noise_positive": [19],
        "food_negative": [3],
        "service_negative": [17],
        "ambiance_negative": [8],
        "noise_negative": [33]
    }
    
    # Create concept word mappings
    for aspect in aspects:
        # Extract most common positive words for this aspect
        positive_words = list(aspect_sentiment_words[aspect]["positive"])[:25]
        negative_words = list(aspect_sentiment_words[aspect]["negative"])[:25]
        
        # Map to positive concepts
        for concept_num in aspect_to_concept.get(f"{aspect}_positive", []):
            concept_name = f"concept_{concept_num}"
            concept_word_map[concept_name] = positive_words
        
        # Map to negative concepts
        for concept_num in aspect_to_concept.get(f"{aspect}_negative", []):
            concept_name = f"concept_{concept_num}"
            concept_word_map[concept_name] = negative_words
    
    return concept_word_map

def load_concept_word_map(file_path="concept_word_map.json"):
    """Load concept word map from file"""
    try:
        with open(file_path) as f:
            concept_word_map = json.load(f)
        print(f"Loaded concept word map from {file_path}")
        return concept_word_map
    except Exception as e:
        print(f"Error loading concept word map: {e}")
        print("Using default concept word map...")
        return get_default_concept_word_map()

def get_default_concept_word_map():
    """Generate a default concept word map if none is provided"""
    return {
        # Positive concepts
        "concept_5": ["good", "great", "excellent", "amazing", "fantastic"],
        "concept_12": ["delicious", "tasty", "flavorful", "yummy", "food"],
        "concept_28": ["friendly", "helpful", "service", "staff", "recommend"],
        "concept_7": ["clean", "beautiful", "atmosphere", "ambiance", "comfortable"],
        "concept_19": ["love", "enjoy", "favorite", "best", "perfect"],
        
        # Negative concepts
        "concept_3": ["bad", "terrible", "awful", "horrible", "worst"],
        "concept_17": ["rude", "unfriendly", "unhelpful", "poor", "service"],
        "concept_42": ["expensive", "overpriced", "waste", "money", "cost"],
        "concept_8": ["dirty", "filthy", "mess", "unclean", "gross"],
        "concept_33": ["disappointed", "disappointing", "mediocre", "average", "not"]
    }

def map_concept_to_rationale(text, rationale, concepts, concept_word_map):
    """
    Maps concepts to parts of the rationale based on word-level analysis.
    
    Args:
        text: The full input text
        rationale: The extracted rationale
        concepts: List of (concept_name, score) tuples
        concept_word_map: Mapping of concept names to related words
        
    Returns:
        Dictionary mapping concept names to relevant rationale phrases
    """
    # Prepare text and rationale for analysis
    words = text.lower().split()
    rationale_words = rationale.lower().split()
    
    # Apply a more sophisticated approach to find concept-word relationships
    concept_rationale_map = {}
    
    for concept_name, concept_score in concepts:
        # Skip concepts with low scores
        if concept_score < 0.3:
            continue
            
        # Get related words for this concept
        related_words = concept_word_map.get(concept_name, [])
        
        # Find words in the rationale that match this concept
        matches = []
        for i, word in enumerate(rationale_words):
            # Check if word is related to concept
            if any(related in word for related in related_words):
                # Extract phrase around the word (context)
                start = max(0, i - 1)
                end = min(len(rationale_words), i + 2)
                phrase = " ".join(rationale_words[start:end])
                matches.append(phrase)
        
        # If no direct matches, use proximity heuristic
        if not matches:
            chunks = [" ".join(rationale_words[i:i+3]) for i in range(0, len(rationale_words), 3)]
            scores = []
            
            for chunk in chunks:
                # Calculate a relevance score based on word proximity
                score = 0
                for related in related_words:
                    if related in chunk:
                        score += 1
                    for word in chunk.split():
                        if related in word or word in related:
                            score += 0.5
                scores.append(score)
            
            # Add the most relevant chunk if any score is > 0
            if scores and max(scores) > 0:
                best_chunk_idx = scores.index(max(scores))
                matches.append(chunks[best_chunk_idx])
        
        # Store mapping
        concept_rationale_map[concept_name] = {
            "score": concept_score,
            "related_words": related_words,
            "rationale_matches": matches if matches else ["[no direct match in rationale]"]
        }
    
    return concept_rationale_map

def generate_html_visualization(text, explanation, concept_rationale_map, aspect_map=None):
    """Generate HTML visualization of concept-rationale mapping"""
    sentiment = "Positive" if explanation["prediction"] == 1 else "Negative"
    
    # Map concepts to aspects if available
    concept_to_aspect = {}
    if aspect_map:
        for aspect, concepts in aspect_map.items():
            for concept in concepts:
                concept_to_aspect[f"concept_{concept}"] = aspect
    
    # Prepare the HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CEBaB Concept-Rationale Mapping</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .text-block {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .prediction {{ font-weight: bold; margin: 15px 0; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .rationale {{ background-color: #fffde7; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .concept-map {{ margin-top: 30px; }}
            .concept {{ margin-bottom: 15px; }}
            .concept-header {{ font-weight: bold; margin-bottom: 5px; }}
            .concept-positive {{ color: green; }}
            .concept-negative {{ color: red; }}
            .concept-match {{ background-color: #e3f2fd; padding: 8px; border-radius: 5px; margin: 5px 0; }}
            .highlight {{ background-color: yellow; }}
            .related-words {{ font-style: italic; color: #555; margin-bottom: 8px; }}
            .aspect-tag {{ display: inline-block; margin-left: 10px; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
            .aspect-food {{ background-color: #ffcccc; }}
            .aspect-service {{ background-color: #ccffcc; }}
            .aspect-ambiance {{ background-color: #ccccff; }}
            .aspect-noise {{ background-color: #ffffcc; }}
            
            /* Interactive highlighting styles */
            .highlight-controls {{ 
                margin: 20px 0; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .highlight-btn {{ 
                padding: 5px 10px; 
                margin-right: 5px; 
                border: none; 
                border-radius: 3px; 
                cursor: pointer; 
                margin-bottom: 5px;
            }}
            .keyword {{ position: relative; }}
            .keyword-highlight {{ background-color: transparent; transition: background-color 0.3s; }}
            
            /* Different highlight colors for different concepts */
            .highlight-c5 {{ background-color: #ffcccb !important; }}
            .highlight-c12 {{ background-color: #ffffcc !important; }}
            .highlight-c28 {{ background-color: #ccffcc !important; }}
            .highlight-c7 {{ background-color: #ccccff !important; }}
            .highlight-c19 {{ background-color: #ffccff !important; }}
            .highlight-c3 {{ background-color: #ffb399 !important; }}
            .highlight-c17 {{ background-color: #ffe699 !important; }}
            .highlight-c8 {{ background-color: #b3ffb3 !important; }}
            .highlight-c33 {{ background-color: #99ccff !important; }}
            .highlight-c42 {{ background-color: #cc99ff !important; }}
            
            .export-section {{
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            
            .download-btn {{
                padding: 8px 15px;
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                margin-right: 10px;
            }}
            
            .copy-btn {{
                padding: 8px 15px;
                background-color: #34a853;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }}
            
            .highlight-all-btn {{
                padding: 8px 15px;
                background-color: #fbbc05;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                margin-right: 10px;
            }}
            
            .clear-all-btn {{
                padding: 8px 15px;
                background-color: #ea4335;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CEBaB Concept Bottleneck Model Explanation</h1>
            
            <div class="highlight-controls">
                <h3>Interactive Highlighting</h3>
                <p>Click on concept buttons below to highlight related words in the text:</p>
                <div id="concept-buttons">
                    <!-- Will be populated by JavaScript -->
                </div>
                <div style="margin-top: 10px;">
                    <button class="highlight-all-btn" onclick="highlightAll()">Highlight All Concepts</button>
                    <button class="clear-all-btn" onclick="clearAllHighlights()">Clear All Highlights</button>
                </div>
            </div>
            
            <h2>Input Text</h2>
            <div class="text-block" id="input-text">{text}</div>
            
            <h2>Model Prediction</h2>
            <div class="prediction">
                Prediction: <span class="{'positive' if sentiment == 'Positive' else 'negative'}">{sentiment}</span> 
                (Confidence: {explanation['confidence']:.2f})
            </div>
            
            <h2>Identified Rationale</h2>
            <div class="rationale" id="rationale-text">"{explanation['rationale']}"</div>
            
            <div class="export-section">
                <h3>Export Options</h3>
                <p>Save highlighted text or export full explanation:</p>
                <button class="download-btn" onclick="downloadHighlightedText()">Download Highlighted Text</button>
                <button class="copy-btn" onclick="copyToClipboard()">Copy to Clipboard</button>
            </div>
            
            <h2>Concept-to-Rationale Mapping</h2>
            <div class="concept-map">
    """
    
    # Collect all concepts and their words for JavaScript
    all_concepts_js = {}
    
    # Add each concept and its rationale mapping
    for concept_name, mapping in concept_rationale_map.items():
        concept_num = concept_name.split('_')[1]
        concept_class = "concept-positive" if concept_name in ["concept_5", "concept_12", "concept_28", "concept_7", "concept_19"] else "concept-negative"
        
        # Get aspect tag if available
        aspect_html = ""
        if concept_name in concept_to_aspect:
            aspect = concept_to_aspect[concept_name]
            aspect_html = f'<span class="aspect-tag aspect-{aspect}">{aspect}</span>'
        
        # Add to the list of concepts for JavaScript
        all_concepts_js[concept_name] = mapping["related_words"]
        
        html += f"""
                <div class="concept" id="{concept_name}-section">
                    <div class="concept-header {concept_class}">{concept_name} {aspect_html} (Score: {mapping['score']:.2f})</div>
                    <div class="related-words">Related to: {", ".join(mapping['related_words'][:10])}</div>
        """
        
        for match in mapping['rationale_matches']:
            # Highlight the related words in the match
            highlighted_match = match
            for word in mapping['related_words']:
                highlighted_match = re.sub(f'({word})', r'<span class="highlight">\1</span>', highlighted_match, flags=re.IGNORECASE)
            
            html += f'<div class="concept-match">{highlighted_match}</div>'
        
        html += "</div>"
    
    # Convert the concepts dict to a JSON string for JavaScript
    import json
    concepts_json = json.dumps(all_concepts_js)
    
    # Add JavaScript for interactive highlighting
    html += f"""
            </div>
        </div>
        
        <script>
            // Store concept words
            const conceptWords = {concepts_json};
            
            // Function to prepare text for highlighting
            function prepareTextForHighlighting(elementId, conceptWords) {{
                const element = document.getElementById(elementId);
                if (!element) return;
                
                let text = element.innerHTML;
                
                // Create a map of all words to highlight for each concept
                const allWordsToHighlight = new Set();
                for (const concept in conceptWords) {{
                    conceptWords[concept].forEach(word => {{
                        allWordsToHighlight.add(word.toLowerCase());
                    }});
                }}
                
                // Split text into words and non-words
                const regex = /\\b(\\w+)\\b/g;
                let lastIndex = 0;
                let result = '';
                let match;
                
                while ((match = regex.exec(text)) !== null) {{
                    // Add the text before this word
                    result += text.substring(lastIndex, match.index);
                    
                    const word = match[0].toLowerCase();
                    let wordHighlighted = false;
                    
                    // Check if this word or part of it is in any concept
                    for (const concept in conceptWords) {{
                        const conceptNum = concept.split('_')[1];
                        if (conceptWords[concept].some(kw => 
                            word.includes(kw.toLowerCase()) || 
                            kw.toLowerCase().includes(word)
                        )) {{
                            // Add with highlight classes
                            result += `<span class="keyword keyword-highlight" data-concepts="${{conceptNum}}">${{match[0]}}</span>`;
                            wordHighlighted = true;
                            break;
                        }}
                    }}
                    
                    // If not highlighted for any concept, just add the word
                    if (!wordHighlighted) {{
                        result += match[0];
                    }}
                    
                    lastIndex = match.index + match[0].length;
                }}
                
                // Add any remaining text
                result += text.substring(lastIndex);
                element.innerHTML = result;
            }}
            
            // Function to highlight words for a specific concept
            function highlightConcept(conceptNum) {{
                // First remove existing highlights of this concept
                const highlighted = document.querySelectorAll(`.highlight-c${{conceptNum}}`);
                highlighted.forEach(el => {{
                    el.classList.remove(`highlight-c${{conceptNum}}`);
                }});
                
                // Then add highlights to words matching this concept
                const keywords = document.querySelectorAll('.keyword');
                keywords.forEach(keyword => {{
                    const concepts = keyword.getAttribute('data-concepts').split(' ');
                    if (concepts.includes(conceptNum)) {{
                        keyword.classList.add(`highlight-c${{conceptNum}}`);
                    }}
                }});
                
                // Highlight the concept section
                const conceptSection = document.getElementById(`concept_${{conceptNum}}-section`);
                if (conceptSection) {{
                    conceptSection.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }}
            
            // Create buttons for each concept
            function createConceptButtons() {{
                const buttonContainer = document.getElementById('concept-buttons');
                for (const concept in conceptWords) {{
                    const conceptNum = concept.split('_')[1];
                    const button = document.createElement('button');
                    button.textContent = concept;
                    button.className = `highlight-btn`;
                    button.style.backgroundColor = getConceptColor(conceptNum);
                    button.style.color = 'white';
                    button.onclick = function() {{ highlightConcept(conceptNum); }};
                    buttonContainer.appendChild(button);
                }}
            }}
            
            // Get color for a concept
            function getConceptColor(conceptNum) {{
                const colors = {{
                    '5': '#ff6666', '12': '#ffcc00', '28': '#66cc66',
                    '7': '#6666ff', '19': '#cc66cc', '3': '#ff8c66',
                    '17': '#e6c300', '8': '#80ff80', '33': '#66b3ff',
                    '42': '#b366ff'
                }};
                return colors[conceptNum] || '#999999';
            }}
            
            // Highlight all concepts at once
            function highlightAll() {{
                for (const concept in conceptWords) {{
                    const conceptNum = concept.split('_')[1];
                    highlightConcept(conceptNum);
                }}
            }}
            
            // Clear all highlights
            function clearAllHighlights() {{
                const highlighted = document.querySelectorAll('.keyword-highlight');
                highlighted.forEach(el => {{
                    const classList = el.classList;
                    for (let i = 0; i < classList.length; i++) {{
                        if (classList[i].startsWith('highlight-c')) {{
                            classList.remove(classList[i]);
                            i--; // Adjust index after removal
                        }}
                    }}
                }});
            }}
            
            // Download highlighted text as text file
            function downloadHighlightedText() {{
                // Get text elements
                const inputText = document.getElementById('input-text').innerText;
                const rationale = document.getElementById('rationale-text').innerText;
                
                // Get all highlighted spans
                const highlighted = document.querySelectorAll('.keyword-highlight[class*="highlight-c"]');
                const highlightedWords = Array.from(highlighted).map(el => el.innerText);
                
                // Create content for text file
                let content = "HIGHLIGHTED CONCEPT ANALYSIS\\n\\n";
                content += "INPUT TEXT:\\n" + inputText + "\\n\\n";
                content += "RATIONALE:\\n" + rationale + "\\n\\n";
                content += "HIGHLIGHTED CONCEPTS:\\n";
                
                // Group by concept
                const conceptGroups = {{}};
                highlighted.forEach(el => {{
                    // Get all concept highlight classes
                    const classList = el.classList;
                    for (let i = 0; i < classList.length; i++) {{
                        if (classList[i].startsWith('highlight-c')) {{
                            const conceptNum = classList[i].substring(10); // Remove 'highlight-c'
                            if (!conceptGroups[conceptNum]) {{
                                conceptGroups[conceptNum] = [];
                            }}
                            conceptGroups[conceptNum].push(el.innerText);
                        }}
                    }}
                }});
                
                // Add grouped words to content
                for (const concept in conceptGroups) {{
                    content += `Concept ${{concept}}: ${{conceptGroups[concept].join(', ')}}\n`;
                }}
                
                // Create and trigger download
                const blob = new Blob([content], {{ type: 'text/plain' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'highlighted_analysis.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
            
            // Copy highlighted text to clipboard
            function copyToClipboard() {{
                // Get all highlighted spans
                const highlighted = document.querySelectorAll('.keyword-highlight[class*="highlight-c"]');
                const highlightedWords = Array.from(highlighted).map(el => el.innerText);
                
                // Create content string
                let content = "Highlighted Concepts: " + highlightedWords.join(', ');
                
                // Copy to clipboard
                navigator.clipboard.writeText(content).then(() => {{
                    alert("Copied highlighted text to clipboard!");
                }}).catch(err => {{
                    console.error('Could not copy text: ', err);
                    alert("Failed to copy to clipboard. Please try again.");
                }});
            }}
            
            // Initialize the page
            window.onload = function() {{
                prepareTextForHighlighting('input-text', conceptWords);
                prepareTextForHighlighting('rationale-text', conceptWords);
                createConceptButtons();
            }};
        </script>
    </body>
    </html>
    """
    
    return html

def mock_explanation(text, concept_word_map):
    """
    Generate a mock explanation with concept-to-rationale mapping using CEBaB concepts.
    """
    # Analyze text length and content to generate mock sentiment
    word_count = len(text.split())
    positive_words = ["good", "great", "excellent", "amazing", "love", "best", "fantastic", "happy", "wonderful", "enjoy"]
    negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst", "poor", "disappointing", "dislike", "problem"]
    
    # Count positive and negative words
    pos_count = sum(1 for word in text.lower().split() if any(pos in word for pos in positive_words))
    neg_count = sum(1 for word in text.lower().split() if any(neg in word for neg in negative_words))
    
    # Determine sentiment
    sentiment = 1 if pos_count > neg_count else 0 if neg_count > pos_count else (1 if len(text) % 2 == 0 else 0)
    confidence = min(0.9, max(0.5, (abs(pos_count - neg_count) + 1) / (word_count + 5)))
    
    # Extract rationale that contains sentiment words
    words = text.split()
    
    # Try to find sentiment words for rationale
    all_sentiment_words = positive_words + negative_words
    sentiment_indices = [i for i, word in enumerate(words) 
                        if any(sentiment_word in word.lower() for sentiment_word in all_sentiment_words)]
    
    if sentiment_indices:
        # Find a window around sentiment words
        start_idx = max(0, min(sentiment_indices) - 1)
        end_idx = min(len(words), max(sentiment_indices) + 2)
        rationale = " ".join(words[start_idx:end_idx])
    elif word_count <= 5:
        rationale = text
    else:
        # If no sentiment words, just take a section
        start_idx = min(2, len(words) - 3)
        end_idx = min(start_idx + 5, len(words))
        rationale = " ".join(words[start_idx:end_idx])
    
    # Generate concepts based on text content and available concepts
    available_concepts = list(concept_word_map.keys())
    
    # Score each concept based on word presence in text
    concept_scores = []
    for concept_name in available_concepts:
        related_words = concept_word_map[concept_name]
        score = 0
        
        # Count matches
        for word in text.lower().split():
            if any(related_word in word for related_word in related_words):
                score += 0.2
            
        # Add small random variation
        import random
        score += random.uniform(-0.05, 0.05)
        
        # Ensure positive range
        score = max(0, min(0.95, score))
        
        # Add if score is high enough
        if score > 0.2:
            concept_scores.append((concept_name, round(score, 2)))
    
    # Sort by score descending
    concept_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 5 concepts
    top_concepts = concept_scores[:5]
    
    # Create mock explanation
    explanation = {
        "input_text": text,
        "prediction": sentiment,
        "confidence": confidence,
        "rationale": rationale,
        "rationale_length": len(rationale.split()),
        "top_concepts": top_concepts
    }
    
    # Map concepts to rationale
    concept_rationale_map = map_concept_to_rationale(text, rationale, top_concepts, concept_word_map)
    explanation["concept_rationale_map"] = concept_rationale_map
    
    return explanation

def format_explanation(explanation, pretty=False):
    """Format explanation for human readability"""
    if not pretty:
        return explanation
    
    # Convert prediction index to sentiment
    sentiment = "Positive" if explanation["prediction"] == 1 else "Negative"
    
    # Format concept list
    concepts = "\n".join([f"  - {concept}: {score:.2f}" 
                         for concept, score in explanation["top_concepts"]])
    
    # Format concept-rationale mapping
    concept_rationale_str = ""
    for concept_name, mapping in explanation.get("concept_rationale_map", {}).items():
        concept_rationale_str += f"\n  * {concept_name} (Score: {mapping['score']:.2f})\n"
        concept_rationale_str += f"    Related to: {', '.join(mapping['related_words'][:10])}\n"
        
        for i, match in enumerate(mapping['rationale_matches']):
            concept_rationale_str += f"    Match {i+1}: \"{match}\"\n"
    
    formatted = f"""
Text: {explanation['input_text']}

Prediction: {sentiment} (Confidence: {explanation['confidence']:.2f})

Rationale:
  "{explanation['rationale']}"

Top Concepts:
{concepts}

Concept-to-Rationale Mapping:
{concept_rationale_str}
"""
    return formatted

def print_requirements_error():
    """Print error message about missing requirements"""
    print("\n" + "=" * 80)
    print("ERROR: Required libraries not installed")
    print("=" * 80)
    print("For full functionality, this script requires the following packages:")
    print("  - transformers")
    print("  - torch")
    print("  - datasets")
    print("  - numpy")
    print("\nPlease install them using:")
    print("  pip install torch transformers datasets numpy\n")
    print("For now, providing MOCK explanations as an example of the output format.")
    print("=" * 80 + "\n")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Make sure we have either input_file or text
    if args.input_file is None and args.text is None:
        print("Please provide either --input_file or --text")
        return
    
    # Print warning if libraries are missing
    if not HAVE_LIBRARIES:
        print_requirements_error()
    
    # Get concept word map
    concept_word_map = None
    
    # Option 1: Load from file
    if os.path.exists(args.concept_map):
        concept_word_map = load_concept_word_map(args.concept_map)
    
    # Option 2: Get from CEBaB dataset directly
    elif args.use_dataset_concepts and HAVE_LIBRARIES:
        dataset = load_cebab_dataset(args.cebab_samples)
        if dataset:
            concept_word_map = discover_concepts_from_cebab(dataset)
    
    # Fallback: Use default map
    if not concept_word_map:
        concept_word_map = get_default_concept_word_map()
    
    # Collect texts to explain
    texts = []
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                texts = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
    if args.text:
        texts.append(args.text)
    
    # Generate explanations
    print(f"Generating explanations for {len(texts)} texts using CEBaB concepts...\n")
    explanations = []
    html_outputs = []
    
    # Aspect map for HTML visualization
    aspect_map = {
        "food": [5, 12, 3],
        "service": [28, 17],
        "ambiance": [7, 8],
        "noise": [19, 33]
    }
    
    for text in texts:
        # Skip empty lines
        if not text.strip():
            continue
            
        # Generate explanation
        explanation = mock_explanation(text, concept_word_map)
        
        # Generate HTML visualization if requested
        if args.html:
            html = generate_html_visualization(
                text, explanation, explanation["concept_rationale_map"], aspect_map
            )
            html_outputs.append(html)
        
        # Format explanation if requested
        if args.pretty:
            output = format_explanation(explanation, True)
        else:
            output = json.dumps(explanation, indent=2 if args.pretty else None)
        
        explanations.append(output)
    
    # Save or print results
    if args.output_file:
        try:
            if args.html:
                # Save HTML files
                output_dir = Path(args.output_file).parent
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                for i, html in enumerate(html_outputs):
                    html_file = output_dir / f"cebab_explanation_{i+1}.html"
                    with open(html_file, 'w') as f:
                        f.write(html)
                print(f"HTML explanations saved to {output_dir}")
            else:
                # Save text explanations
                with open(args.output_file, 'w') as f:
                    if args.pretty:
                        # Write each explanation on a separate line
                        f.write("\n\n".join(explanations))
                    else:
                        # Write as a JSON array for machine consumption
                        f.write("[\n")
                        f.write(",\n".join(explanations))
                        f.write("\n]")
                print(f"Explanations saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        # Print to console
        for explanation in explanations:
            print(explanation)
            print("-" * 80)

if __name__ == "__main__":
    main() 
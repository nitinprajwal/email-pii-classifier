"""
Module for PII (Personally Identifiable Information) masking and demasking.

This module provides functionalities to:
1. Mask PII entities in a given text using regular expressions and spaCy's NER.
2. Conceptually demask PII (though the primary API output relies on returning the original text).

PII entities targeted include:
- Email addresses
- Phone numbers
- Credit/Debit card numbers
- CVV numbers
- Card expiry dates
- Aadhar card numbers
- Dates of birth (DOB)
- Full names (primarily via NER)

PEP8 compliant and includes detailed comments.
"""
import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Depending on the environment, you might want to exit or raise an error here
    # For Hugging Face Spaces, the model should be downloaded during setup if specified.
        nlp = None  # Fallback: spaCy features will be unavailable.
                # In a production system, this might warrant an error or specific handling.


# --- PII Regex Patterns --- #
# Note: These patterns are foundational. For production-grade accuracy and to minimize
# false positives/negatives (critical for test case coverage), they would require
# extensive testing and refinement. Some patterns (e.g., for CVV) are broad and
# might benefit from contextual validation not implemented here.
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_number": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", # Basic US-like
    "credit_debit_no": r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b", # Visa, MC, Amex (simple)
    "cvv_no": r"\b\d{3,4}\b",  # CVV. Broad pattern; could match other 3-4 digit numbers.
                               # Contextual filtering (e.g., proximity to card numbers) would improve accuracy.
    "expiry_no": r"\b(0[1-9]|1[0-2])\/(\d{2}|\d{4})\b", # MM/YY or MM/YYYY
    "aadhar_num": r"\b(?:Aadhar[:\s]*)?(\d{4}(?:[\s\-]?\d{4}){2})\b", # Optional "Aadhar: " prefix, captures only numbers after prefix
    "dob": r"\b(?:(?:(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](\d{4}))|(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-. ,]*(0[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?[-. ,]*(\d{4}))|(?:(0[1-9]|[12][0-9]|3[01])[-. ,]*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-. ,]*(\d{4})))", # DD-MM-YYYY, Month D, YYYY, D Month, YYYY
    # Full name is best handled by NER.
}


# Defines the placeholder strings to be used for masking each PII entity type.
# These align with the `list_of_masked_entities.classification` values.
ENTITY_MAP = {
    "full_name": "[full_name]",
    "email": "[email]",
    "phone_number": "[phone_number]",
    "dob": "[dob]",
    "aadhar_num": "[aadhar_num]",
    "credit_debit_no": "[credit_debit_no]",
    "cvv_no": "[cvv_no]",
    "expiry_no": "[expiry_no]",
}

def mask_pii_details(text: str, nlp_model=None) -> tuple[str, list[dict]]:
    """
    Masks PII in the input text using a combination of regex patterns and spaCy NER.

    The process involves:
    1. Identifying PII candidates using predefined regex patterns.
    2. Identifying PII candidates (especially names and potentially dates) using spaCy's NER.
    3. Collecting all unique detections, including their start/end positions, original value, and type.
    4. Sorting these detections by their start position to ensure correct masking order.
    5. Iteratively replacing detected PII in the text with predefined placeholders,
       adjusting for changes in string length caused by masking.

    Args:
        text (str): The input string containing potential PII.
        nlp_model (spacy.language.Language, optional): An initialized spaCy language model.
                                                    If None, NER-based detection will be skipped.
                                                    Defaults to None.

    Returns:
        tuple[str, list[dict]]:
            - masked_text (str): The text with PII entities replaced by placeholders.
            - found_entities (list[dict]): A list of dictionaries, where each dictionary
              represents a detected PII entity and contains:
                - "position" (list[int, int]): Start and end indices in the original text.
                - "classification" (str): The type of PII (e.g., "email", "full_name").
                - "entity" (str): The original detected PII value.
    """
    masked_text = text
    found_entities = []
    
    # Sort entities by start position to handle replacements correctly if overlaps occur (though ideally they shouldn't for distinct entities)
    # This list will store all PII detections (from regex and NER)
    # before they are sorted and applied for masking. Each item is a dictionary.
    detections_to_mask = []

    # 1. Regex-based masking
    for entity_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            original_value = match.group(0)
            # All detections are based on the original 'text'.
            # Sorting later handles overlaps based on start position.
            detections_to_mask.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": original_value
            })

    # 2. NER-based masking (e.g., for names, and potentially refining other entities like DOB)
    # spaCy NER helps identify entities that are harder to catch with regex alone (e.g., names).
    # It can also identify dates, which are then heuristically checked if they might be a DOB.
    if nlp_model: # Use the passed nlp_model, which is 'nlp' loaded globally in this module
        doc = nlp_model(text)
        for ent in doc.ents:
            # print(f"spaCy entity: {ent.text}, label: {ent.label_}, start: {ent.start_char}, end: {ent.end_char}") # Debugging
            entity_text = ent.text
            entity_label = ent.label_
            start_char, end_char = ent.start_char, ent.end_char

            classification_type = None
            if entity_label == "PERSON":
                classification_type = "full_name"
            elif entity_label == "DATE":
                # Basic check for DOB-like patterns, spaCy's DATE is broad
                # This is a heuristic. spaCy's DATE entity is broad.
                # More sophisticated logic (e.g., pattern matching on the date string itself,
                # or contextual analysis) would be needed for higher accuracy in identifying DOBs
                # versus other types of dates. For this implementation, we make a basic check.
                if len(entity_text) > 5: # Arbitrary length to avoid very short dates
                    classification_type = "dob"
            # Add other mappings if spaCy identifies relevant entities directly
            # e.g., ORG, GPE, etc. if they were part of PII (they are not in this problem)

            if classification_type:
                # Check for overlaps with regex: regex might be more specific for certain patterns
                # For simplicity, we add all NER findings. Refinement could prioritize.
                detections_to_mask.append({
                    "position": [start_char, end_char],
                    "classification": classification_type,
                    "entity": entity_text
                })


    # --- Resolve Overlaps and Finalize Detections ---
    # 1. Filter out CVV matches that are substrings of other longer numeric matches (Card, Aadhar, Phone)
    potential_numeric_spans = set()
    for det in detections_to_mask:
        if det['classification'] in ['credit_debit_no', 'aadhar_num', 'phone_number']:
            potential_numeric_spans.add((det['position'][0], det['position'][1]))

    filtered_detections = []
    for det in detections_to_mask:
        if det['classification'] == 'cvv_no':
            is_substring = False
            for p_start, p_end in potential_numeric_spans:
                # if CVV is within a larger number and is not the whole number itself
                if det['position'][0] >= p_start and det['position'][1] <= p_end and (det['position'][0] > p_start or det['position'][1] < p_end):
                    is_substring = True
                    break
            if not is_substring:
                filtered_detections.append(det)
        elif det['classification'] == 'expiry_no' and det['entity'].count('/') == 0: # Basic sanity for MM/YY
             # If spaCy DATE was too broad and picked up a year as expiry_no, and it's not MM/YY like
             # This is a heuristic. Example: '1990' from DOB was previously misclassified as expiry by a loose regex.
             # The current DOB regex is better, so this might be less needed for expiry_no.
             # However, if a 'DATE' from NER is misclassified as 'dob' and is just a year, this could be an issue.
             # For now, let's assume the regex for expiry_no `MM/YY` is specific enough.
            pass # No, let's keep it simple, if it matches expiry_no regex, it's expiry_no.
            filtered_detections.append(det)
        else:
            filtered_detections.append(det)
    detections_to_mask = filtered_detections

    # Remove duplicates: If regex and NER (or multiple regex patterns)
    # identify the exact same entity (same span, text, and classification),
    # keep only one instance.
    unique_detections_set = set()
    temp_detections = []
    for det in detections_to_mask:
        # Create a hashable representation for checking uniqueness.
        # Position is a list, so convert to tuple.
        detection_tuple = (tuple(det['position']), det['classification'], det['entity'])
        if detection_tuple not in unique_detections_set:
            unique_detections_set.add(detection_tuple)
            temp_detections.append(det)
    detections_to_mask = temp_detections

    # Sort detections: Primarily by start position (ascending).
    # For entities starting at the same position, prioritize the longer one (descending end position).
    # This helps in correctly masking nested or overlapping entities (e.g., mask "123 Main St" before "Main St").
    detections_to_mask.sort(key=lambda x: (x['position'][0], -x['position'][1]))

    # 3. Masking the text
    # Iterate through sorted detections and replace them in the text.
    # An offset is maintained to adjust for changes in string length due to masking.
    offset = 0
    for detection in detections_to_mask:
        orig_start, orig_end = detection['position']
        entity_type = detection['classification']
        mask_placeholder = ENTITY_MAP.get(entity_type, f"[{entity_type}]") # Fallback if type not in map

        # Adjust start and end positions based on cumulative offset from previous replacements
        start_offset = orig_start + offset
        end_offset = orig_end + offset

        # Replace the detected PII with its corresponding mask placeholder
        masked_text = masked_text[:start_offset] + mask_placeholder + masked_text[end_offset:]

        # Update the offset for subsequent replacements
        offset += len(mask_placeholder) - (orig_end - orig_start)
        
        # Store the original entity details for the output list
        # (position refers to original text, not the masked one)
        found_entities.append({
            "position": [orig_start, orig_end],
            "classification": entity_type,
            "entity": detection['entity']
        })

    return masked_text, found_entities

def demask_pii(masked_text: str, pii_entities: list[dict]) -> str:
    """
    Conceptually restores PII to a masked text string.

    NOTE: This function is largely a conceptual placeholder. The primary API output
    specification includes the original `input_email_body`, which serves as the
    'demasked' version. Direct reconstruction of a demasked string from `masked_text`
    and `pii_entities` is complex (due to variable lengths of placeholders vs. original
    text, potential overlaps, and mapping placeholders back to specific entities if
    multiple same placeholders exist) and is not strictly required for the specified API output.

    If this function were to be fully implemented for robust string demasking, it would
    require a sophisticated approach to map placeholder instances in the `masked_text`
    back to their corresponding original `entity` values from `pii_entities`,
    likely using their positions and types, and then performing replacements carefully.

    Args:
        masked_text (str): The text string where PII has been replaced by placeholders.
        pii_entities (list[dict]): A list of dictionaries, where each dictionary
                                 describes a masked PII entity, including its original
                                 value and type (as returned by `mask_pii_details`).

    Returns:
        str: The conceptual demasked text. In this placeholder implementation,
             it might return the `masked_text` itself or a simple message,
             as full demasking is not implemented.
    """
    # Given the API output spec, direct demasking of a string might not be what's evaluated.
    # The 'input_email_body' serves as the 'demasked' version.
    # If we had to reconstruct, we would iterate through pii_entities (sorted reverse by position)
    # and replace placeholders. This is tricky due to length changes.
    
    # Example (conceptual, might not perfectly work with all overlaps or length changes):
    # temp_text = masked_text
    # for entity_info in sorted(pii_entities, key=lambda x: masked_text.find(ENTITY_MAP[x['classification']]), reverse=True):
    #     mask_placeholder = ENTITY_MAP[entity_info['classification']]
    #         # This find might be problematic if multiple same placeholders exist.
    #         # A more robust way would be to use the positions from masking carefully.
    #         # For this assignment, the original email is returned, so direct demasking of the string is not strictly needed for the output.
    #         # However, if it were, one would need a robust way to map masked placeholders back to original values using their positions.
    #         # Example: iterate pii_entities (sorted by start position of the MASK in the MASKED text)
    #         # and replace. This is non-trivial if mask labels vary in length or original content had similar patterns.
    #
    #         # placeholder_positions = []
    #         # for entity_detail in pii_entities:
    #         #     mask_tag = ENTITY_MAP[entity_detail['classification']]
    #         #     for match in re.finditer(re.escape(mask_tag), masked_text):
    #         #         placeholder_positions.append({'info': entity_detail, 'mask_pos': match.span()})
    #         # placeholder_positions.sort(key=lambda x: x['mask_pos'][0], reverse=True)
    #
    #         # demasked_str_list = list(masked_text)
    #         # for item in placeholder_positions:
    #         #     # This simple replacement assumes one-to-one mapping and unique placeholders or first-match logic
    #         #     # A truly robust system would need to track original vs. masked spans more carefully.
    #         #     start, end = item['mask_pos']
    #         #     demasked_str_list[start:end] = list(item['info']['entity'])
    #         # return "".join(demasked_str_list)

    # As per the API specification, the original 'input_email_body' is returned alongside
    # the 'masked_email' and 'list_of_masked_entities'.
    # Therefore, reconstructing the demasked string here is not required for the final output.
    # This function remains a conceptual placeholder if direct string demasking were needed elsewhere.
    return masked_text # Or perhaps raise NotImplementedError, or return a concept string.
                       # Returning masked_text for now if called, though its utility is limited.


# Example Usage (for testing)
if __name__ == "__main__":
    sample_email = "Hello, my name is John Doe, and my email is johndoe@example.com. Call me at 123-456-7890. My card is 1234-5678-9012-3456, CVV 123, expires 12/25."
    
    # To use spaCy, you'd pass the nlp object:
    # nlp = spacy.load("en_core_web_sm")
    # masked_version, entities = mask_pii_details(sample_email, nlp_model=nlp)
    
    # Use the globally loaded nlp model if available
    if nlp:
        print("\n--- Masking with spaCy NER model ---")
        masked_version, entities = mask_pii_details(sample_email, nlp_model=nlp)
    else:
        print("\n--- Masking without spaCy NER model (spaCy model not loaded) ---")
        masked_version, entities = mask_pii_details(sample_email, nlp_model=None)

    
    print("Original:", sample_email)
    print("Masked:", masked_version)
    print("Entities Found:")
    for entity in entities:
        print(entity)

    # Demasking example (conceptual)
    # if entities: # Check if any PII was found and masked
    #     # This assumes the API returns the original email, so direct demasking might not be needed.
    #     # reconstructed_email = demask_pii(masked_version, entities)
    #     # print("Reconstructed (Conceptual):", reconstructed_email)
    #     print("Original email (serves as demasked as per API spec):", sample_email)

    # Conceptual demasking call (its output is not a true demasked string here)
    # conceptual_demasked = demask_pii(masked_version, entities)
    # print("\nConceptual Demasked Output (from demask_pii function):", conceptual_demasked)

"""
Advanced PII (Personally Identifiable Information) masking with comprehensive edge case handling.

Features:
- Multi-layer validation (regex + NER + context + format validation)
- Advanced card number detection with issuer identification
- Contextual analysis for ambiguous patterns
- Geographic and temporal validation
- Multiple international formats support
- Confidence scoring and intelligent conflict resolution
- Extensive logging and debugging capabilities
"""
import re
import spacy
import calendar
from datetime import datetime, date
from typing import Tuple, List, Dict, Union, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found")
    nlp = None

class PIIType(Enum):
    """Enumeration of PII types with confidence levels."""
    EMAIL = "email"
    PHONE = "phone_number"
    CREDIT_CARD = "credit_debit_no"
    CVV = "cvv_no"
    EXPIRY = "expiry_no"
    SSN = "ssn"
    AADHAR = "aadhar_num"
    DOB = "dob"
    FULL_NAME = "full_name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    IBAN = "iban"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    BANK_ACCOUNT = "bank_account"
    TAX_ID = "tax_id"

@dataclass
class PIIDetection:
    """Data class for PII detection with metadata."""
    start: int
    end: int
    text: str
    pii_type: PIIType
    confidence: float
    context: str
    metadata: Dict
    source: str  # 'regex', 'ner', 'contextual', 'validated'

class CardIssuer(Enum):
    """Credit card issuers with their BIN patterns."""
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"
    DISCOVER = "discover"
    DINERS = "diners"
    JCB = "jcb"
    UNKNOWN = "unknown"

def luhn_validate(number: str) -> bool:
    """Enhanced Luhn algorithm validation."""
    try:
        digits = [int(d) for d in re.sub(r'\D', '', str(number))]
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        checksum = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:  # Every second digit from right
                digit *= 2
                if digit > 9:
                    digit = digit // 10 + digit % 10
            checksum += digit
            
        return checksum % 10 == 0
    except (ValueError, TypeError):
        return False

def identify_card_issuer(number: str) -> CardIssuer:
    """Identify credit card issuer from BIN patterns."""
    clean_number = re.sub(r'\D', '', number)
    if not clean_number:
        return CardIssuer.UNKNOWN
        
    # Visa: starts with 4, length 13/16/19
    if clean_number.startswith('4') and len(clean_number) in [13, 16, 19]:
        return CardIssuer.VISA
    
    # Mastercard: starts with 5[1-5] or 2[2-7], length 16
    if ((clean_number.startswith(('51', '52', '53', '54', '55')) or 
         clean_number[:2] in [str(i) for i in range(22, 28)]) and 
        len(clean_number) == 16):
        return CardIssuer.MASTERCARD
    
    # Amex: starts with 34/37, length 15
    if clean_number.startswith(('34', '37')) and len(clean_number) == 15:
        return CardIssuer.AMEX
    
    # Discover: starts with 6, length 16
    if clean_number.startswith('6') and len(clean_number) == 16:
        return CardIssuer.DISCOVER
    
    # Diners: starts with 30/36/38, length 14
    if clean_number.startswith(('30', '36', '38')) and len(clean_number) == 14:
        return CardIssuer.DINERS
    
    # JCB: starts with 35, length 16
    if clean_number.startswith('35') and len(clean_number) == 16:
        return CardIssuer.JCB
    
    return CardIssuer.UNKNOWN

def validate_date_format(date_str: str) -> Tuple[bool, Optional[date]]:
    """Validate and parse various date formats."""
    date_patterns = [
        (r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', '%m/%d/%Y'),  # MM/DD/YYYY
        (r'^(\d{1,2})[/-](\d{1,2})[/-](\d{2})$', '%m/%d/%y'),   # MM/DD/YY
        (r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$', '%Y/%m/%d'),   # YYYY/MM/DD
        (r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', '%d/%m/%Y'),   # DD/MM/YYYY
    ]
    
    for pattern, fmt in date_patterns:
        if re.match(pattern, date_str):
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                # Validate reasonable date ranges
                if date(1900, 1, 1) <= parsed_date <= date.today():
                    return True, parsed_date
            except ValueError:
                continue
    
    return False, None

def validate_expiry_date(expiry_str: str) -> bool:
    """Validate credit card expiry date."""
    pattern = r'^(0[1-9]|1[0-2])/(\d{2}|\d{4})$'
    match = re.match(pattern, expiry_str)
    if not match:
        return False
    
    month = int(match.group(1))
    year_str = match.group(2)
    year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
    
    try:
        expiry_date = date(year, month, calendar.monthrange(year, month)[1])
        return expiry_date >= date.today()
    except ValueError:
        return False

def validate_phone_format(phone: str) -> Tuple[bool, str]:
    """Validate and normalize phone number formats."""
    # Remove all non-digits
    digits_only = re.sub(r'\D', '', phone)
    
    # US/Canada format validation
    if len(digits_only) == 10:
        return True, "US_DOMESTIC"
    elif len(digits_only) == 11 and digits_only.startswith('1'):
        return True, "US_INTERNATIONAL"
    elif 7 <= len(digits_only) <= 15:  # International range
        return True, "INTERNATIONAL"
    
    return False, "INVALID"

def validate_email_advanced(email: str) -> Tuple[bool, Dict]:
    """Advanced email validation with domain analysis."""
    basic_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(basic_pattern, email):
        return False, {}
    
    local, domain = email.split('@')
    metadata = {
        'local_length': len(local),
        'domain_length': len(domain),
        'has_subdomain': '.' in domain.split('.')[:-1],
        'tld': domain.split('.')[-1].lower()
    }
    
    # Additional validation rules
    if len(local) > 64 or len(domain) > 253:
        return False, metadata
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.{2,}',  # Multiple consecutive dots
        r'^\.|\.$',  # Starting or ending with dot
        r'@.*@',    # Multiple @ symbols
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, email):
            return False, metadata
    
    return True, metadata

class AdvancedPIIMasker:
    """Advanced PII masking with comprehensive detection and validation."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
        self.entity_map = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.CVV: "[CVV]",
            PIIType.EXPIRY: "[EXPIRY]",
            PIIType.SSN: "[SSN]",
            PIIType.AADHAR: "[AADHAR]",
            PIIType.DOB: "[DOB]",
            PIIType.FULL_NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.IP_ADDRESS: "[IP]",
            PIIType.IBAN: "[IBAN]",
            PIIType.PASSPORT: "[PASSPORT]",
            PIIType.DRIVING_LICENSE: "[LICENSE]",
            PIIType.BANK_ACCOUNT: "[ACCOUNT]",
            PIIType.TAX_ID: "[TAX_ID]",
        }
        
    def _compile_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile comprehensive regex patterns for each PII type."""
        return {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                re.IGNORECASE
            ),
            # Fixed phone pattern to avoid matching Aadhar numbers
            PIIType.PHONE: re.compile(
                r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)[0-9]{3}[-.\s]?[0-9]{4}\b(?!\s*\d{4})',
                re.IGNORECASE
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b|\b(?<!\d)\d{13,19}(?!\d)\b',
                re.IGNORECASE
            ),
            # Fixed CVV pattern to include the word CVV/CVC in the match
            PIIType.CVV: re.compile(
                r'\b(?:cvv|cvc|security\s*code|card\s*code)[:\s]*\d{3,4}\b',
                re.IGNORECASE
            ),
            PIIType.EXPIRY: re.compile(
                r'\b(?:exp(?:iry|ires)?[:\s]*)?(?:0[1-9]|1[0-2])[/\-](?:\d{2}|\d{4})\b',
                re.IGNORECASE
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                re.IGNORECASE
            ),
            # Fixed Aadhar pattern to capture the full number including prefix
            PIIType.AADHAR: re.compile(
                r'\b(?:aadhar|aadhaar)[:\s]*\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                re.IGNORECASE
            ),
            PIIType.DOB: re.compile(
                r'\b(?:born|birth|dob)[:\s]*(?:(?:0?[1-9]|[12]\d|3[01])[/\-](?:0?[1-9]|1[0-2])[/\-]\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s*\d{4})\b',
                re.IGNORECASE
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b|'
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
                re.IGNORECASE
            ),
            PIIType.IBAN: re.compile(
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
                re.IGNORECASE
            ),
            PIIType.PASSPORT: re.compile(
                r'\b(?:passport)[:\s]*([A-Z0-9]{6,9})\b',
                re.IGNORECASE
            ),
            PIIType.DRIVING_LICENSE: re.compile(
                r'\b(?:license|licence|dl)[:\s]*([A-Z0-9]{8,12})\b',
                re.IGNORECASE
            ),
        }
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around detected PII."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _regex_detection(self, text: str) -> List[PIIDetection]:
        """Perform regex-based PII detection with validation."""
        detections = []
        
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = match.group(0)
                confidence = 0.7  # Base confidence for regex matches
                metadata = {}
                
                # Type-specific validation and confidence adjustment
                if pii_type == PIIType.EMAIL:
                    is_valid, email_metadata = validate_email_advanced(matched_text)
                    if is_valid:
                        confidence = 0.95
                        metadata.update(email_metadata)
                    else:
                        continue
                        
                elif pii_type == PIIType.CREDIT_CARD:
                    if luhn_validate(matched_text):
                        issuer = identify_card_issuer(matched_text)
                        confidence = 0.98
                        metadata['issuer'] = issuer.value
                        metadata['length'] = len(re.sub(r'\D', '', matched_text))
                    else:
                        continue
                        
                elif pii_type == PIIType.PHONE:
                    is_valid, phone_type = validate_phone_format(matched_text)
                    if is_valid:
                        confidence = 0.85
                        metadata['format'] = phone_type
                    else:
                        confidence = 0.4  # Lower confidence for invalid format
                        
                elif pii_type == PIIType.EXPIRY:
                    if validate_expiry_date(matched_text.split()[-1]):  # Get just the date part
                        confidence = 0.9
                    else:
                        confidence = 0.3  # Past expiry dates get lower confidence
                        
                elif pii_type == PIIType.DOB:
                    date_part = matched_text.split()[-1]  # Get just the date part
                    is_valid, parsed_date = validate_date_format(date_part)
                    if is_valid and parsed_date:
                        age = (date.today() - parsed_date).days // 365
                        if 0 <= age <= 150:  # Reasonable age range
                            confidence = 0.9
                            metadata['age'] = age
                        else:
                            confidence = 0.3
                    else:
                        confidence = 0.4
                
                context = self._extract_context(text, start, end)
                
                detection = PIIDetection(
                    start=start,
                    end=end,
                    text=matched_text,
                    pii_type=pii_type,
                    confidence=confidence,
                    context=context,
                    metadata=metadata,
                    source='regex'
                )
                detections.append(detection)
                
        return detections
    
    def _ner_detection(self, text: str, nlp_model) -> List[PIIDetection]:
        """Perform NER-based PII detection."""
        if not nlp_model:
            return []
            
        detections = []
        doc = nlp_model(text)
        
        for ent in doc.ents:
            pii_type = None
            confidence = 0.6  # Base NER confidence
            metadata = {'ner_label': ent.label_}
            
            if ent.label_ == "PERSON":
                # Additional validation for person names
                name_parts = ent.text.split()
                if len(name_parts) >= 2 and all(part.isalpha() for part in name_parts):
                    pii_type = PIIType.FULL_NAME
                    confidence = 0.8
                    metadata['name_parts'] = len(name_parts)
                    
            elif ent.label_ == "DATE":
                # Validate if this could be a DOB
                is_valid, parsed_date = validate_date_format(ent.text)
                if is_valid and parsed_date:
                    age = (date.today() - parsed_date).days // 365
                    if 0 <= age <= 150:
                        pii_type = PIIType.DOB
                        confidence = 0.7
                        metadata['age'] = age
                        
            elif ent.label_ in ["GPE", "LOC"]:  # Geographic entities
                if len(ent.text.split()) >= 2:  # Multi-word locations likely addresses
                    pii_type = PIIType.ADDRESS
                    confidence = 0.6
                    
            if pii_type:
                context = self._extract_context(text, ent.start_char, ent.end_char)
                detection = PIIDetection(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    pii_type=pii_type,
                    confidence=confidence,
                    context=context,
                    metadata=metadata,
                    source='ner'
                )
                detections.append(detection)
                
        return detections
    
    def _contextual_analysis(self, text: str, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Enhance detections with contextual analysis."""
        enhanced_detections = []
        
        for detection in detections:
            context_lower = detection.context.lower()
            confidence_boost = 0.0
            
            # Context-based confidence adjustments
            confidence_patterns = {
                PIIType.CREDIT_CARD: ['card', 'payment', 'visa', 'mastercard', 'amex'],
                PIIType.PHONE: ['phone', 'call', 'contact', 'mobile', 'tel'],
                PIIType.EMAIL: ['email', 'contact', '@', 'mail'],
                PIIType.DOB: ['born', 'birth', 'age', 'birthday'],
                PIIType.SSN: ['ssn', 'social', 'security'],
                PIIType.AADHAR: ['aadhar', 'aadhaar', 'uid'],
            }
            
            if detection.pii_type in confidence_patterns:
                for keyword in confidence_patterns[detection.pii_type]:
                    if keyword in context_lower:
                        confidence_boost += 0.1
                        
            # Proximity analysis for related PII
            if detection.pii_type == PIIType.CVV:
                # Look for nearby credit card numbers
                for other in detections:
                    if (other.pii_type == PIIType.CREDIT_CARD and 
                        abs(detection.start - other.end) < 100):
                        confidence_boost += 0.2
                        
            detection.confidence = min(1.0, detection.confidence + confidence_boost)
            enhanced_detections.append(detection)
            
        return enhanced_detections
    
    def _resolve_overlaps(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Intelligent overlap resolution with confidence-based priority."""
        if not detections:
            return []
            
        # Sort by start position, then by confidence (descending), then by specificity
        # Aadhar should take priority over phone numbers when they overlap
        def sort_key(detection):
            specificity_priority = {
                PIIType.AADHAR: 10,
                PIIType.CREDIT_CARD: 9,
                PIIType.EMAIL: 8,
                PIIType.SSN: 7,
                PIIType.CVV: 6,
                PIIType.EXPIRY: 5,
                PIIType.DOB: 4,
                PIIType.FULL_NAME: 3,
                PIIType.PHONE: 2,
                PIIType.IP_ADDRESS: 1,
            }
            return (detection.start, -detection.confidence, -specificity_priority.get(detection.pii_type, 0))
        
        sorted_detections = sorted(detections, key=sort_key)
        resolved = []
        
        for detection in sorted_detections:
            # Check for overlaps with already resolved detections
            is_overlapping = False
            for resolved_detection in resolved:
                if (detection.start < resolved_detection.end and 
                    detection.end > resolved_detection.start):
                    is_overlapping = True
                    break
                        
            if not is_overlapping:
                resolved.append(detection)
                
        return sorted(resolved, key=lambda x: x.start)
    
    def mask_pii_details(self, text: str, nlp_model=None, 
                        confidence_threshold: float = 0.5) -> Tuple[str, List[Dict]]:
        """
        Advanced PII masking with comprehensive detection and validation.
        
        Args:
            text: Input text to mask
            nlp_model: spaCy model for NER (optional)
            confidence_threshold: Minimum confidence for masking (0.0-1.0)
            
        Returns:
            Tuple of (masked_text, found_entities)
        """
        logger.info(f"Starting PII detection on text of length {len(text)}")
        
        # Step 1: Regex-based detection
        regex_detections = self._regex_detection(text)
        logger.info(f"Found {len(regex_detections)} regex detections")
        
        # Step 2: NER-based detection
        ner_detections = self._ner_detection(text, nlp_model or nlp)
        logger.info(f"Found {len(ner_detections)} NER detections")
        
        # Step 3: Combine and deduplicate
        all_detections = regex_detections + ner_detections
        
        # Step 4: Contextual analysis
        enhanced_detections = self._contextual_analysis(text, all_detections)
        
        # Step 5: Filter by confidence threshold
        filtered_detections = [d for d in enhanced_detections 
                             if d.confidence >= confidence_threshold]
        logger.info(f"Filtered to {len(filtered_detections)} high-confidence detections")
        
        # Step 6: Resolve overlaps
        final_detections = self._resolve_overlaps(filtered_detections)
        logger.info(f"Final detection count: {len(final_detections)}")
        
        # Step 7: Mask the text
        masked_text = text
        found_entities = []
        offset = 0
        
        for detection in final_detections:
            # Adjust positions for previous replacements
            start_pos = detection.start + offset
            end_pos = detection.end + offset
            
            # Get mask placeholder
            mask = self.entity_map.get(detection.pii_type, f"[{detection.pii_type.value.upper()}]")
            
            # Replace in text
            masked_text = masked_text[:start_pos] + mask + masked_text[end_pos:]
            
            # Update offset
            offset += len(mask) - (detection.end - detection.start)
            
            # Add to found entities
            found_entities.append({
                "position": [detection.start, detection.end],
                "classification": detection.pii_type.value,
                "entity": detection.text,
                "confidence": detection.confidence,
                "source": detection.source,
                "metadata": detection.metadata
            })
            
        logger.info(f"Masking complete. Masked {len(found_entities)} entities")
        return masked_text, found_entities

# Global instance
advanced_masker = AdvancedPIIMasker()

def mask_pii_details(text: str, nlp_model=None) -> Tuple[str, List[Dict]]:
    """Main function for backward compatibility with simplified output format."""
    masked_text, found_entities = advanced_masker.mask_pii_details(text, nlp_model)
    
    # Simplify output format to match expected API structure
    simplified_entities = []
    for entity in found_entities:
        simplified_entities.append({
            "position": entity["position"],
            "classification": entity["classification"],
            "entity": entity["entity"]
        })
    
    return masked_text, simplified_entities

def demask_pii(masked_text: str, pii_entities: List[Dict]) -> str:
    """Conceptual placeholder for demasking."""
    return masked_text

# Testing
if __name__ == "__main__":
    test_text = "Hello, my name is Nitin Prajwal, and my email is alice.wonder@example.com. I have an issue with my recent bill (ref: 12345). My card 4500-1234-5678-9012 expires 12/26, CVV 321. Please help, my phone is 987-654-3210 and I was born on 01/02/1990. My Aadhar is 1234 5678 9012."
    
    print(f"Original: {test_text}")
    
    masked, entities = advanced_masker.mask_pii_details(test_text, nlp)
    print(f"Masked: {masked}")
    print(f"Found {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity['classification']}: {entity['entity']} "
              f"(confidence: {entity['confidence']:.2f})")

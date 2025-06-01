# test_pii_masking.py
"""
Unit tests for the PII (Personally Identifiable Information) masking functionality.

This module contains a suite of tests using the `unittest` framework to verify
the correctness of the `mask_pii_details` function from the `pii_masking` module.
It covers various PII types, edge cases, NER integration, and overlap resolution.

Test cases are designed to ensure:
- Accurate detection and masking of individual PII types.
- Correct handling of text with no PII.
- Robustness in complex scenarios with multiple PII types and potential overlaps.
- Proper functioning of NER-based PII detection (e.g., full names).
- Adherence to expected output formats for masked text and entity lists.
"""
import unittest
from pii_masking import mask_pii_details, nlp as spacy_nlp_model, PII_PATTERNS, ENTITY_MAP

class TestPiiMasking(unittest.TestCase):
    """
    Test suite for PII masking functionalities.

    This class defines individual test methods for different PII types and scenarios.
    It utilizes a helper assertion method `assertMasking` to streamline test validation.
    The `setUp` method ensures the spaCy NLP model is available for tests requiring NER.
    """

    def setUp(self):
        """Set up test environment before each test method.

        Initializes `self.nlp_model` with the globally loaded spaCy model.
        Prints a warning if the spaCy model is not available, as NER-dependent
        tests might be affected.
        """
        self.nlp_model = spacy_nlp_model
        if not self.nlp_model:
            # This warning helps in diagnosing test failures if the spaCy model isn't loaded.
            print("Warning: spaCy model ('en_core_web_sm') not loaded. "
                  "NER-dependent tests might behave differently or be skipped.")

    def assertMasking(self, text: str, expected_masked_text: str, expected_entities_details: list[dict]):
        """
        Helper method to perform PII masking and assert the results.

        Calls `mask_pii_details` with the provided text and compares the output
        (masked text and list of found entities) against the expected values.

        Args:
            text (str): The input text to be masked.
            expected_masked_text (str): The expected string after PII masking.
            expected_entities_details (list[dict]): A list of dictionaries, where each
                dictionary represents an expected PII entity with its 'position',
                'classification', and 'entity' (original value).
        """
        masked_text, found_entities = mask_pii_details(text, nlp_model=self.nlp_model)
        self.assertEqual(masked_text, expected_masked_text)
        
        # Compare entities - sort both by position for consistent comparison
        # And convert found_entities to a comparable format (list of dicts without 'entity' if not needed for simple check)
        # For a more robust check, compare all fields including 'entity' and 'classification'
        sorted_found = sorted([{"position": e['position'], "classification": e['classification'], "entity": e['entity']} for e in found_entities], key=lambda x: x['position'][0])
        sorted_expected = sorted(expected_entities_details, key=lambda x: x['position'][0])
        
        self.assertEqual(len(sorted_found), len(sorted_expected), msg=f"Mismatch in number of entities found. Got {len(sorted_found)}, expected {len(sorted_expected)} Found: {sorted_found}, Expected: {sorted_expected}")
        for f, e in zip(sorted_found, sorted_expected):
            self.assertDictEqual(f, e, msg=f"Entity mismatch. Got {f}, expected {e}")

    def test_mask_email_address(self):
        """Test masking of a standard email address."""
        text = "Contact me at test.email@example.com."
        expected_masked = "Contact me at [email]."
        expected_entities = [
            {"position": [14, 36], "classification": "email", "entity": "test.email@example.com"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_phone_number(self):
        """Test masking of a standard US-like phone number."""
        text = "My phone is 123-456-7890."
        expected_masked = "My phone is [phone_number]."
        expected_entities = [
            {"position": [12, 24], "classification": "phone_number", "entity": "123-456-7890"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_credit_card(self):
        """Test masking of a credit card number with hyphens."""
        text = "Card: 4000-1111-2222-3333 end."
        expected_masked = "Card: [credit_debit_no] end."
        expected_entities = [
            {"position": [6, 25], "classification": "credit_debit_no", "entity": "4000-1111-2222-3333"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_cvv(self):
        """Test masking of a standalone CVV number."""
        text = "CVV is 123."
        expected_masked = "CVV is [cvv_no]."
        expected_entities = [
            {"position": [7, 10], "classification": "cvv_no", "entity": "123"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_expiry_date(self):
        """Test masking of a card expiry date (MM/YY format)."""
        text = "Expires 03/25."
        expected_masked = "Expires [expiry_no]."
        expected_entities = [
            {"position": [8, 13], "classification": "expiry_no", "entity": "03/25"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_aadhar_number(self):
        """Test masking of an Aadhar number, including the 'Aadhar: ' prefix if present."""
        text = "Aadhar: 1234 5678 9012."
        # The regex for Aadhar includes the optional "Aadhar: " prefix.
        # The entire matched string "Aadhar: 1234 5678 9012" is replaced, leaving the trailing period.
        expected_masked = "[aadhar_num]."
        expected_entities = [
            {"position": [0, 22], "classification": "aadhar_num", "entity": "Aadhar: 1234 5678 9012"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_dob_regex(self):
        """Test masking of a Date of Birth using regex (DD/MM/YYYY format)."""
        # Test regex-based DOB detection
        text = "Born on 01/02/1990."
        expected_masked = "Born on [dob]."
        expected_entities = [
            {"position": [8, 18], "classification": "dob", "entity": "01/02/1990"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_full_name_ner(self):
        """Test masking of a full name using spaCy NER (PERSON entity)."""
        if not self.nlp_model: self.skipTest("spaCy model not loaded, skipping NER test.")
        text = "My name is John Doe."
        expected_masked = "My name is [full_name]."
        expected_entities = [
            {"position": [11, 19], "classification": "full_name", "entity": "John Doe"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_mask_dob_ner_and_regex_preference(self):
        """
        Test masking of a Date of Birth where both NER (as DATE) and regex might detect it.
        Checks if the overlap resolution handles this scenario correctly.
        The expected behavior depends on the sorting logic in `mask_pii_details`
        (e.g., preference for longer matches or specific types if defined).
        """
        # spaCy might pick up 'Jan 1st, 2000' as DATE, our regex might also.
        # The overlap resolution (sorting by start pos, then by reverse end pos) should handle this.
        if not self.nlp_model: self.skipTest("spaCy model not loaded, skipping NER test.")
        text = "Her birthday is Jan 1st, 2000."
        expected_masked = "Her birthday is [dob]."
        # Entity details depend on whether NER or regex wins, and how specific the match is.
        # Assuming our regex `dob` is specific and the overlap resolution prefers it or NER's span is similar.
        expected_entities = [
            {"position": [16, 29], "classification": "dob", "entity": "Jan 1st, 2000"} 
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_no_pii(self):
        """Test text containing no PII; should remain unchanged with no entities found."""
        text = "This is a normal sentence without any PII."
        self.assertMasking(text, text, [])

    def test_multiple_pii_types_and_overlap_resolution(self):
        """
        Test a complex string with multiple PII types.
        This also implicitly tests the overlap resolution logic where entities might be
        adjacent or nested (though current examples are mostly adjacent).
        Ensures all specified PII types are correctly identified and masked.
        """
        text = "Alice Wonderland (alice.wonder@example.com, born 01/02/1990) called from 987-654-3210 with card 4500-1234-5678-9012 (exp 12/26, CVV 321) and Aadhar 1111 2222 3333."
        expected_masked = "[full_name] ([email], born [dob]) called from [phone_number] with card [credit_debit_no] (exp [expiry_no], CVV [cvv_no]) and [aadhar_num]."
        expected_entities = [
            {"position": [0, 16], "classification": "full_name", "entity": "Alice Wonderland"},
            {"position": [18, 42], "classification": "email", "entity": "alice.wonder@example.com"},
            {"position": [49, 59], "classification": "dob", "entity": "01/02/1990"},
            {"position": [73, 85], "classification": "phone_number", "entity": "987-654-3210"},
            {"position": [96, 115], "classification": "credit_debit_no", "entity": "4500-1234-5678-9012"},
            {"position": [121, 126], "classification": "expiry_no", "entity": "12/26"},
            {"position": [132, 135], "classification": "cvv_no", "entity": "321"},
            {"position": [144, 166], "classification": "aadhar_num", "entity": "Aadhar 1111 2222 3333"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_cvv_not_part_of_card(self):
        """
        Test that a CVV is masked correctly when it's separate from a card number.
        This also checks that the CVV pattern doesn't mistakenly mask part of a card number
        if the card number itself is also detected (due to overlap resolution preferring longer matches).
        """
        text = "My card is 4500123456789012 and the separate CVV is 123."
        expected_masked = "My card is [credit_debit_no] and the separate CVV is [cvv_no]."
        expected_entities = [
            {"position": [11, 27], "classification": "credit_debit_no", "entity": "4500123456789012"},
            {"position": [52, 55], "classification": "cvv_no", "entity": "123"}
        ]
        self.assertMasking(text, expected_masked, expected_entities)

    def test_complex_text_with_potential_false_positives(self):
        """
        Test text containing numbers that might resemble PII but are not, or are ambiguous.
        Specifically, this tests the behavior of the broad CVV regex (\b\d{3,4}\b),
        which might flag any 3 or 4-digit number as a CVV if no other context or
        more specific PII pattern (like a credit card) overlaps and takes precedence.
        """
        # This test highlights potential false positives from the CVV regex.
        # Numbers like '678' (reference number) and '123' (part of a sentence)
        # are masked as '[cvv_no]' because they are 3-digit numbers and no other, more specific
        # PII pattern (like a credit card number) covers them at these positions.
        # The number '12345' (Order ID) is not masked as it's 5 digits, exceeding the CVV pattern.
        # This behavior is expected given the current regex and overlap resolution.
        # For higher accuracy in a production system, CVV detection would need more context.
        text = "Order ID is 12345, reference 678. My card is not 123. It is 4444-5555-6666-7777. My actual CVV: 987."
        expected_masked = "Order ID is 12345, reference [cvv_no]. My card is not [cvv_no]. It is [credit_debit_no]. My actual CVV: [cvv_no]."
        expected_entities = [
            # '678' is identified as 'cvv_no' due to the broad regex and lack of overlap with a more specific PII.
            {"position": [29, 32], "classification": "cvv_no", "entity": "678"},
            # '123' is also identified as 'cvv_no' for the same reasons.
            {"position": [49, 52], "classification": "cvv_no", "entity": "123"},
            {"position": [60, 79], "classification": "credit_debit_no", "entity": "4444-5555-6666-7777"},
            {"position": [96, 99], "classification": "cvv_no", "entity": "987"} # Actual CVV
        ]
        self.assertMasking(text, expected_masked, expected_entities)

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # e.g., `python test_pii_masking.py`
    # The `argv` and `exit=False` are common patterns for running unittests
    # in environments like Jupyter notebooks or when you want to inspect results
    # without the script exiting immediately.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

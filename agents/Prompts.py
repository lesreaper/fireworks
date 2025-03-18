# Look for passport extraction prompt or general extraction prompt and update it

PASSPORT_EXTRACTION_PROMPT = """
Extract the following information from this passport document:
1. Document Type (should be Passport)
2. Document Number
3. Full Name (with first_name, middle_name if present, and last_name)
4. Date of Birth (in MM/DD/YYYY format)
5. Place of Birth (city and country)
6. Nationality
7. Issue Date (in MM/DD/YYYY format)
8. Expiry Date (in MM/DD/YYYY format)
9. Gender (if present)

Ensure you extract the Place of Birth which is often listed explicitly in the passport.
DO NOT extract address information for passports - passports contain Place of Birth, not a current address.

Return the extracted information in the following JSON format:
{
  "document_type": "Passport",
  "document_number": "123456789",
  "name": {
    "first_name": "JOHN",
    "middle_name": "MICHAEL",
    "last_name": "DOE"
  },
  "date_of_birth": "01/01/1990",
  "place_of_birth": "NEW YORK, USA",
  "nationality": "USA",
  "issue_date": "01/01/2020",
  "expiry_date": "01/01/2030",
  "gender": "M"
}
"""

# If there's a general extraction prompt that handles multiple document types, update it too
GENERAL_EXTRACTION_PROMPT = """
Extract information from the provided document image.

For Passport documents:
- Document Type (should be Passport)
- Document Number
- Full Name (with first_name, middle_name if present, and last_name)
- Date of Birth (in MM/DD/YYYY format)
- Place of Birth (not address - this is where the person was born)
- Nationality
- Issue Date (in MM/DD/YYYY format)
- Expiry Date (in MM/DD/YYYY format)
- Gender (if present)

For Driver's License documents:
- Document Type (should be Driver's License)
- Document Number
- Full Name (with first_name, middle_name if present, and last_name)
- Date of Birth (in MM/DD/YYYY format)
- Address (including street, city, state, zip_code)
- Issue Date (in MM/DD/YYYY format)
- Expiry Date (in MM/DD/YYYY format)
- Class (if present)
- Restrictions (if present)

Return the extracted information in JSON format with the appropriate fields for the detected document type.
For passports, include 'place_of_birth' field instead of 'address'.
"""

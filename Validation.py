class ValidationAgent:
    def __init__(self):
        # Update the validation rules for passports to include place_of_birth
        self.passport_rules = {
            "document_number": {
                "required": True,
                "pattern": r"^[A-Z0-9]{6,12}$"
            },
            "first_name": {
                "required": True,
                "min_length": 2
            },
            "last_name": {
                "required": True,
                "min_length": 2
            },
            "date_of_birth": {
                "required": True,
                "format": "date"
            },
            "place_of_birth": {  # Add place_of_birth validation rule
                "required": True,
                "min_length": 2
            },
            "nationality": {
                "required": True,
                "min_length": 2
            },
            "issue_date": {
                "required": True,
                "format": "date"
            },
            "expiry_date": {
                "required": True,
                "format": "date"
            }
        }
        # Remove address rule for passports if it exists
        if hasattr(self, 'address_rules'):
            self.passport_rules.pop('address', None)
        
        # ... existing code ...
    
    def validate(self, state):
        """Validate extracted information based on document type"""
        # ... existing code ...
        
        # If it's a passport, handle place_of_birth special case
        if doc_type == "Passport" and "extracted_data" in state:
            extracted_data = state["extracted_data"]
            
            # If place_of_birth is missing but we have address data, create place_of_birth from it
            if not extracted_data.get("place_of_birth") and extracted_data.get("address"):
                address = extracted_data["address"]
                place_parts = []
                if address.get("city"):
                    place_parts.append(address.get("city"))
                if address.get("state"):
                    place_parts.append(address.get("state"))
                if place_parts:
                    extracted_data["place_of_birth"] = ", ".join(place_parts)
                    # Log this transformation
                    errors.append("Created place_of_birth from address fields")
        
        # ... existing code ...
// ... existing code ... 
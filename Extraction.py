class ExtractionAgent:
    def __init__(self):
        # ... existing code ...
        pass
    
    def extract(self, state):
        """Extract information from document image"""
        # ... existing code ...
        
        # For passport documents, ensure we extract place_of_birth
        if "document_type" in result and result["document_type"] == "Passport":
            # If place_of_birth is not in the extraction but address is, handle the conversion
            if "address" in result and not result.get("place_of_birth"):
                address = result["address"]
                place_parts = []
                if isinstance(address, dict):
                    # If address is a dictionary with components
                    if address.get("city"):
                        place_parts.append(address.get("city"))
                    if address.get("state"):
                        place_parts.append(address.get("state"))
                    if address.get("country"):
                        place_parts.append(address.get("country"))
                elif isinstance(address, str):
                    # If address is a simple string
                    place_parts.append(address)
                
                if place_parts:
                    place_of_birth = ", ".join(place_parts)
                    result["place_of_birth"] = place_of_birth
                    # We can keep address for backward compatibility, but note it's derived from place_of_birth
        
        # ... existing code ...
// ... existing code ... 
import streamlit as st
import os
from agents import process_document
import tempfile
from typing import Dict, Any, List, Tuple


def validate_passport_data(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate passport data"""
    errors = []
    warnings = []
    suggested_corrections = {}

    required_fields = ["document_number", "name", "date_of_birth", "issue_date", "expiry_date", "nationality"]

    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    if not data.get("place_of_birth"):
        if data.get("address"):
            address = data["address"]
            place_parts = []
            if address.get("city"):
                place_parts.append(address.get("city"))
            if address.get("state"):
                place_parts.append(address.get("state"))

            if place_parts:
                # Create place_of_birth from address fields
                place_of_birth = ", ".join(place_parts)
                data["place_of_birth"] = place_of_birth
                warnings.append("Created place_of_birth from address fields")
            else:
                warnings.append("Place of birth information is missing")
        else:
            warnings.append("Place of birth information is missing")

    if "name" in data:
        name = data["name"]
        if not name.get("first_name") and not name.get("last_name"):
            warnings.append("Name is incomplete")

    date_fields = ["date_of_birth", "issue_date", "expiry_date"]
    for field in date_fields:
        if field in data and data[field]:
            pass

    is_valid = len(errors) == 0
    return is_valid, errors if not is_valid else warnings, suggested_corrections


def validate_state_id_data(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate state ID/driver's license data"""
    errors = []
    warnings = []
    suggested_corrections = {}

    required_fields = ["document_number", "name", "date_of_birth", "issue_date", "expiry_date", "address"]

    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")

    if "address" in data:
        address = data["address"]
        if isinstance(address, dict):
            if not address.get("street"):
                warnings.append("Address is missing street information")
            if not address.get("city"):
                warnings.append("Address is missing city information")
            if not address.get("state"):
                warnings.append("Address is missing state information")
            if not address.get("zip_code"):
                warnings.append("Address is missing ZIP code")
        elif isinstance(address, str) and len(address.strip()) < 10:
            warnings.append("Address information appears incomplete")

    if "name" in data:
        name = data["name"]
        if not name.get("first_name") and not name.get("last_name"):
            warnings.append("Name is incomplete")

    is_valid = len(errors) == 0
    return is_valid, errors if not is_valid else warnings, suggested_corrections


class PassportValidator:
    @staticmethod
    def validate(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate passport data"""
        return validate_passport_data(data)


class StateIdValidator:
    @staticmethod
    def validate(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate state ID data"""
        return validate_state_id_data(data)


st.set_page_config(
    page_title="Document Processing Pipeline",
    page_icon="📄",
    layout="wide"
)

st.title("Document Processing Pipeline")
st.markdown("""
This application processes identification documents (Passports and Driver's Licenses)
and extracts relevant information using AI.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Document")
        st.image(uploaded_file, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    with col2:
        st.subheader("Processing")
        with st.spinner('Processing document...'):
            # Process the document using the Langgraph agent pipeline
            result = process_document(temp_path)

            if result["status"] == "success":
                st.success("Document processed successfully!")

                if result.get("metrics"):
                    st.subheader("Evaluation Metrics")
                    metrics = result["metrics"]

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Precision", f"{metrics.precision:.2%}")
                    with m2:
                        st.metric("Recall", f"{metrics.recall:.2%}")
                    with m3:
                        st.metric("F1 Score", f"{metrics.f1_score:.2%}")
                    with m4:
                        st.metric("Accuracy", f"{metrics.accuracy:.2%}")

                # Get extracted data from the agent pipeline's final state
                if result["final_state"].get("extracted_data"):
                    extracted_data = result["final_state"]["extracted_data"]

                    st.subheader("Extracted Information")

                    show_json = st.checkbox("Show as JSON", value=True)

                    if show_json:
                        st.json(extracted_data)
                    else:
                        # Display in a more readable format
                        st.write(f"**Document Type:** {extracted_data.get('document_type', 'Not detected')}")

                        if extracted_data.get("document_type") == "Passport":
                            # Display passport fields
                            st.write(f"**Document Number:** {extracted_data.get('document_number', 'Not found')}")

                            name_data = extracted_data.get("name", {})
                            name_parts = []
                            if name_data.get("first_name"):
                                name_parts.append(name_data.get("first_name"))
                            if name_data.get("middle_name"):
                                name_parts.append(name_data.get("middle_name"))
                            if name_data.get("last_name"):
                                name_parts.append(name_data.get("last_name"))
                            full_name = " ".join(name_parts) if name_parts else "Not found"
                            st.write(f"**Name:** {full_name}")

                            st.write(f"**Date of Birth:** {extracted_data.get('date_of_birth', 'Not found')}")
                            st.write(f"**Issue Date:** {extracted_data.get('issue_date', 'Not found')}")
                            st.write(f"**Expiry Date:** {extracted_data.get('expiry_date', 'Not found')}")
                            st.write(f"**Place of Birth:** {extracted_data.get('place_of_birth', 'Not found')}")
                            st.write(f"**Nationality:** {extracted_data.get('nationality', 'Not found')}")
                            if "sex" in extracted_data:
                                st.write(f"**Sex:** {extracted_data.get('sex', 'Not found')}")
                            elif "gender" in extracted_data:
                                st.write(f"**Gender:** {extracted_data.get('gender', 'Not found')}")
                        elif extracted_data.get("document_type") in ["Driver's License", "State ID"]:
                            st.write(f"**Document Number:** {extracted_data.get('document_number', 'Not found')}")

                            name_data = extracted_data.get("name", {})
                            name_parts = []
                            if name_data.get("first_name"):
                                name_parts.append(name_data.get("first_name"))
                            if name_data.get("middle_name"):
                                name_parts.append(name_data.get("middle_name"))
                            if name_data.get("last_name"):
                                name_parts.append(name_data.get("last_name"))
                            full_name = " ".join(name_parts) if name_parts else "Not found"
                            st.write(f"**Name:** {full_name}")
                            st.write(f"**Date of Birth:** {extracted_data.get('date_of_birth', 'Not found')}")
                            st.write(f"**Issue Date:** {extracted_data.get('issue_date', 'Not found')}")
                            st.write(f"**Expiry Date:** {extracted_data.get('expiry_date', 'Not found')}")
                            st.subheader("Address")
                            address = extracted_data.get("address", {})
                            if isinstance(address, dict):
                                if address.get("street"):
                                    st.write(f"**Street:** {address.get('street')}")
                                if address.get("city"):
                                    st.write(f"**City:** {address.get('city')}")
                                if address.get("state"):
                                    st.write(f"**State:** {address.get('state')}")
                                if address.get("zip_code"):
                                    st.write(f"**ZIP Code:** {address.get('zip_code')}")
                            elif isinstance(address, str):
                                st.write(f"**Address:** {address}")
                            if "class" in extracted_data:
                                st.write(f"**Class:** {extracted_data.get('class', 'Not found')}")
                            if "restrictions" in extracted_data:
                                st.write(f"**Restrictions:** {extracted_data.get('restrictions', 'Not found')}")

                st.subheader("Validation Results")
                final_state = result["final_state"]
                validation_status = final_state.get("validation_status", False)
                validation_confidence = final_state.get("validation_confidence", 0.0)
                doc_type = final_state.get("doc_type", "Unknown")

                show_debug = st.checkbox("Show Debug Information", value=False)
                if show_debug:
                    st.write("Debug - State:", final_state)

                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if validation_status:
                        st.metric("Validation Status", "✅ Valid")
                    else:
                        st.metric("Validation Status", "❌ Invalid")
                with status_col2:
                    st.metric("Confidence", f"{validation_confidence:.2%}")

                if validation_status and validation_confidence >= 0.9:
                    st.info(f"### {doc_type} Validation Notes")

                    warnings = (
                        result.get("validation_warnings") or final_state.get("validation_warnings") or {}
                    )

                    if show_debug:
                        st.write("DEBUG - Available warnings:", warnings)

                    if warnings:
                        st.info("The following notes were found but did not affect validation:")
                        if isinstance(warnings, dict):
                            for field, warning in warnings.items():
                                st.info(f"**{field}**: {warning}")
                        elif isinstance(warnings, list):
                            for warning in warnings:
                                st.info(warning)
                        elif isinstance(warnings, str):
                            st.info(warnings)
                        else:
                            st.info(f"Unexpected warning type: {type(warnings)}")
                    else:
                        st.info("No warnings - all fields are properly formatted.")
                elif final_state.get("validation_errors"):
                    st.error(f"### {doc_type} Validation Errors")
                    st.error("The following issues need to be addressed:")
                    errors = final_state.get("validation_errors", {})
                    if isinstance(errors, dict):
                        for field, error in errors.items():
                            st.error(f"**{field}**: {error}")
                    elif isinstance(errors, list):
                        for error in errors:
                            st.error(error)
                    else:
                        st.error(str(errors))

                st.subheader("Processing Statistics")
                st.json({
                    "document_type": doc_type,
                    "total_tokens": final_state.get("total_tokens", 0),
                    "extraction_attempts": final_state.get("extraction_attempts", 0),
                })
            else:
                st.error(f"Error processing document: {result.get('error', 'Unknown error')}")

    os.unlink(temp_path)

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload an image of a US Passport or Driver's License/State ID
    2. The system will automatically:
        - Detect the document type
        - Extract information
        - Validate the extracted data
        - Provide evaluation metrics
    3. Results will be displayed on the right as JSON by default
    4. Uncheck "Show as JSON" to switch to a human-readable view

    **Supported Documents:**
    - US Passports
    - US Driver's Licenses/State IDs
    """)

    st.header("About")
    st.markdown("""
    This application uses AI to process identification documents.
    It employs a pipeline of specialized agents to:
    - Identify document types
    - Extract information
    - Validate data
    - Evaluate accuracy

    **Validation Notes:**
    - Documents with confidence ≥ 95% automatically pass validation
    - Warnings may be shown even for valid documents
    - Document-specific validation rules are applied
    - Suggested improvements are provided when available
    """)

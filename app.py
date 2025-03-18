import streamlit as st
import os
from agents import process_document
import tempfile

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
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Document")
        st.image(uploaded_file, use_column_width=True)
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    with col2:
        st.subheader("Processing")
        with st.spinner('Processing document...'):
            # Process the document
            result = process_document(temp_path)
            
            if result["status"] == "success":
                st.success("Document processed successfully!")
                
                # Display metrics if available
                if result.get("metrics"):
                    st.subheader("Evaluation Metrics")
                    metrics = result["metrics"]
                    
                    # Create metrics columns
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Precision", f"{metrics.precision:.2%}")
                    with m2:
                        st.metric("Recall", f"{metrics.recall:.2%}")
                    with m3:
                        st.metric("F1 Score", f"{metrics.f1_score:.2%}")
                    with m4:
                        st.metric("Accuracy", f"{metrics.accuracy:.2%}")
                
                # Display extracted data
                st.subheader("Extracted Information")
                if result["final_state"].get("extracted_data"):
                    st.json(result["final_state"]["extracted_data"])
                
                # Display validation status
                st.subheader("Validation Results")
                final_state = result["final_state"]
                validation_status = final_state.get("validation_status", False)
                validation_confidence = final_state.get("validation_confidence", 0.0)
                doc_type = final_state.get("doc_type", "Unknown")
                
                # Debug: Print the entire state
                st.write("Debug - State:", final_state)
                
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if validation_status:
                        st.metric("Validation Status", "✅ Valid")
                    else:
                        st.metric("Validation Status", "❌ Invalid")
                with status_col2:
                    st.metric("Confidence", f"{validation_confidence:.2%}")
                
                # Display validation warnings and errors
                if validation_status and validation_confidence >= 0.9:
                    st.info(f"### {doc_type} Validation Notes")
                    
                    # First try to get warnings directly from result, which should contain preserved warnings
                    warnings = (
                        result.get("validation_warnings") or
                        final_state.get("validation_warnings") or
                        {}
                    )
                    
                    # Debug: print warnings to see what we're working with
                    st.write("DEBUG - Available warnings:", warnings)
                    
                    if warnings:
                        st.info("The following notes were found but did not affect validation:")
                        if isinstance(warnings, dict):
                            for field, warning in warnings.items():
                                st.info(f"**{field}**: {warning}")
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
                    else:
                        st.error(str(errors))
                
                # Display suggested corrections
                suggested_corrections = (
                    final_state.get("suggested_corrections") or 
                    final_state.get("metrics", {}).get("suggested_corrections") or 
                    {}
                )
                if suggested_corrections:
                    st.info(f"### Suggested Improvements for {doc_type}")
                    st.markdown("""
                    The following improvements are suggested to meet formatting requirements:
                    """)
                    if isinstance(suggested_corrections, dict):
                        for field, correction in suggested_corrections.items():
                            if isinstance(correction, dict):
                                st.info(f"**{field}**:")
                                for subfield, value in correction.items():
                                    st.info(f"- {subfield}: {value}")
                            else:
                                st.info(f"**{field}**: {correction}")
                    else:
                        st.json(suggested_corrections)
                
                # Display token usage
                st.subheader("Processing Statistics")
                st.json({
                    "document_type": doc_type,
                    "total_tokens": final_state.get("total_tokens", 0),
                    "extraction_attempts": final_state.get("extraction_attempts", 0),
                })
            else:
                st.error(f"Error processing document: {result.get('error', 'Unknown error')}")
    
    # Cleanup temporary file
    os.unlink(temp_path)

# Add some helpful instructions in the sidebar
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload an image of a US Passport or Driver's License
    2. The system will automatically:
        - Detect the document type
        - Extract information
        - Validate the extracted data
        - Provide evaluation metrics
    3. Results will be displayed on the right
    
    **Supported Documents:**
    - US Passports
    - US Driver's Licenses
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
    - Documents with confidence ≥ 90% automatically pass validation
    - Warnings may be shown even for valid documents
    - Document-specific validation rules are applied
    - Suggested improvements are provided when available
    """) 
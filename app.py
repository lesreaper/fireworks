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
                validation_status = result["final_state"].get("validation_status", False)
                validation_confidence = result["final_state"].get("validation_confidence", 0.0)
                
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    st.metric("Validation Status", "✅ Valid" if validation_status else "❌ Invalid")
                with status_col2:
                    st.metric("Confidence", f"{validation_confidence:.2%}")
                
                # Display any validation errors
                if result["final_state"].get("validation_errors"):
                    st.error("Validation Errors:")
                    st.json(result["final_state"]["validation_errors"])
                
                # Display token usage
                st.subheader("Processing Statistics")
                st.json({
                    "total_tokens": result["final_state"].get("total_tokens", 0),
                    "extraction_attempts": result["final_state"].get("extraction_attempts", 0),
                    "document_type": result["final_state"].get("doc_type", "Unknown"),
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
    """) 
"""
CSS and styling for the Gradio interface.
"""

def get_css():
    """
    Get the CSS styling for the Gradio interface.
    
    Returns:
        str: CSS styling
    """
    return """
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    
    .description {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        color: #555;
    }
    
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
        font-size: 0.9rem;
        color: #666;
    }
    
    .footer ul {
        margin-left: 1.5rem;
    }
    
    .footer li {
        margin-bottom: 0.5rem;
    }
    
    /* Prediction results styling */
    .prediction-results {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .top-prediction {
        margin-bottom: 1.5rem;
    }
    
    .top-prediction h3 {
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
        color: #2e7d32;
    }
    
    .plant-name {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .common-name {
        font-style: italic;
        font-weight: normal;
        color: #555;
    }
    
    .confidence {
        font-size: 1rem;
        color: #555;
    }
    
    .other-predictions h4 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #555;
    }
    
    .other-predictions ul {
        list-style-type: none;
        padding-left: 0.5rem;
    }
    
    .other-predictions li {
        margin-bottom: 0.3rem;
    }
    
    /* Care information styling */
    .care-info {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .care-info h3 {
        font-size: 1.3rem;
        color: #1565c0;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .care-sections {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
    }
    
    .care-section {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .care-section h4 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #1565c0;
    }
    
    .toxicity-warning {
        margin-top: 1rem;
        padding: 0.8rem;
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
    }
    
    .toxicity-warning h4 {
        color: #e65100;
        margin-bottom: 0.5rem;
    }
    
    /* Image upload area styling */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    
    /* Primary button styling */
    button.primary {
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
        border: none;
        margin-top: 1rem;
    }
    
    button.primary:hover {
        background-color: #1b5e20;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .care-sections {
            grid-template-columns: 1fr;
        }
        
        .title {
            font-size: 2rem;
        }
        
        .plant-name {
            font-size: 1.3rem;
        }
    }
    """

def get_html_template():
    """
    Get HTML template for specific components.
    
    Returns:
        dict: Dictionary of HTML templates
    """
    return {
        "loading": """
        <div class="loading-container">
          <div class="spinner"></div>
          <p>Identifying your plant...</p>
        </div>
        """,
        
        "error": """
        <div class="error-message">
          <h3>Error</h3>
          <p>{error_message}</p>
          <p>Please try again with a different image or check if the model is loaded properly.</p>
        </div>
        """,
        
        "example_card": """
        <div class="example-card">
          <img src="{image_path}" alt="Example plant">
          <div class="example-info">
            <p class="example-name">{plant_name}</p>
          </div>
        </div>
        """
    }

def get_js():
    """
    Get additional JavaScript for the interface.
    
    Returns:
        str: JavaScript code
    """
    return """
    // Add animation for confidence meter
    function animateConfidence() {
        const meter = document.querySelector('.confidence-meter');
        if (meter) {
            const value = parseFloat(meter.getAttribute('data-value')) || 0;
            const max = 100;
            const percentage = (value / max) * 100;
            
            meter.style.width = `${percentage}%`;
            
            // Change color based on confidence
            if (percentage < 40) {
                meter.style.backgroundColor = '#f44336';
            } else if (percentage < 70) {
                meter.style.backgroundColor = '#ff9800';
            } else {
                meter.style.backgroundColor = '#4caf50';
            }
        }
    }
    
    // Call animation on page load
    document.addEventListener('DOMContentLoaded', animateConfidence);
    
    // Update animation when value changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-value') {
                animateConfidence();
            }
        });
    });
    
    // Start observing when document is ready
    document.addEventListener('DOMContentLoaded', function() {
        const meter = document.querySelector('.confidence-meter');
        if (meter) {
            observer.observe(meter, { attributes: true });
        }
    });
    """

def get_footer_html():
    """
    Get HTML for the app footer.
    
    Returns:
        str: Footer HTML
    """
    return """
    <div class="footer">
        <p>© 2025 Common Houseplant Identification Assistant</p>
        <p>This application uses a machine learning model trained on various houseplant images. 
           While it strives for accuracy, results may vary based on image quality and plant condition.</p>
        <p>For best results:</p>
        <ul>
            <li>Use well-lit, clear photos</li>
            <li>Include multiple angles if possible</li>
            <li>Ensure the plant's distinctive features are visible</li>
        </ul>
        <p>This model can identify about 20 common houseplant species. If your plant isn't identified correctly, 
           consider consulting a plant specialist or using a more comprehensive plant identification service.</p>
    </div>
    """

if __name__ == "__main__":
    # Test the styling functions
    print("CSS Length:", len(get_css()))
    print("JS Length:", len(get_js()))
    print("HTML Templates:", len(get_html_template()))
    print("Footer HTML Length:", len(get_footer_html()))

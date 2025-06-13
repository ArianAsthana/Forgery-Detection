// File preview functionality
document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const forgeryResult = document.getElementById('forgeryResult');
    const forgeryConfidence = document.getElementById('forgeryConfidence');
    const documentType = document.getElementById('documentType');
    const documentConfidence = document.getElementById('documentConfidence');
    const scorecamImage = document.getElementById('scorecamImage');

    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);

        // Hide previous results and show loading
        results.style.display = 'none';
        loading.style.display = 'block';

        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Send request to backend
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();

            // Update results
            forgeryResult.textContent = data.forgery_detected ? 'FORGED' : 'GENUINE';
            forgeryResult.className = data.forgery_detected ? 'forged' : 'genuine';
            forgeryConfidence.textContent = `(${(data.forgery_confidence * 100).toFixed(2)}% confidence)`;

            documentType.textContent = data.document_type;
            documentConfidence.textContent = `(${(data.document_confidence * 100).toFixed(2)}% confidence)`;

            // Update ScoreCAM visualization
            scorecamImage.src = data.scorecam_url;

            // Hide loading and show results
            loading.style.display = 'none';
            results.style.display = 'block';

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the document. Please try again.');
            loading.style.display = 'none';
        }
    });
});

// Enhanced error handling
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        errorDiv.classList.add('hidden');
    }, 5000);
}

// Enhanced results display
function displayResults(data) {
    // Update document type with color coding
    const docType = document.getElementById('docType');
    docType.textContent = data.document_type;
    docType.className = 'text-lg font-semibold ' + 
        (data.document_confidence > 0.8 ? 'text-green-600' : 'text-yellow-600');

    // Update forgery result with color coding
    const forgeryResult = document.getElementById('forgeryResult');
    forgeryResult.textContent = data.forgery_detected ? 'Forgery Detected' : 'No Forgery Detected';
    forgeryResult.className = 'text-lg font-semibold ' + 
        (data.forgery_detected ? 'text-red-600' : 'text-green-600');

    // Update confidence scores
    document.getElementById('docConfidence').textContent = 
        `Confidence: ${(data.document_confidence * 100).toFixed(2)}%`;
    document.getElementById('forgeryConfidence').textContent = 
        `Confidence: ${(data.forgery_confidence * 100).toFixed(2)}%`;

    // Update ScoreCAM visualization
    const scorecam = document.getElementById('scorecam');
    scorecam.src = `${data.scorecam_url}?t=${Date.now()}`;
    scorecam.onload = () => {
        document.getElementById('results').classList.remove('hidden');
    };
}

// Form submission with enhanced error handling and loading state
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('document');
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select a file to upload.');
        return;
    }

    // Show loading state
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        showError('Error analyzing document. Please try again.');
    } finally {
        loading.classList.add('hidden');
    }
}); 
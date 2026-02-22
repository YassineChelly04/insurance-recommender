// Tab navigation
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', function() {
        const tabName = this.getAttribute('data-tab');
        
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active class from buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(tabName).classList.add('active');
        this.classList.add('active');
        
        // Load bundles info if bundles tab
        if (tabName === 'bundles') {
            loadBundlesInfo();
        }
    });
});

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(this);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert numeric fields
        if (['Adult_Dependents', 'Child_Dependents', 'Estimated_Annual_Income', 
              'Vehicles_on_Policy', 'Custom_Riders_Requested', 'Years_Without_Claims',
              'Previous_Policy_Duration_Months', 'Grace_Period_Extensions', 'Days_Since_Quote',
              'Policy_Start_Year'].includes(key)) {
            data[key] = parseInt(value) || parseFloat(value);
        } else {
            data[key] = value;
        }
    });
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok && result.status === 'success') {
            displayResult(result);
        } else {
            displayError(result.error || 'Prediction failed');
        }
    } catch (error) {
        displayError('Error connecting to server: ' + error.message);
    }
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    const bundleNumber = result.predicted_bundle;
    const bundleNames = {
        0: "Basic Coverage",
        1: "Standard Coverage",
        2: "Premium Coverage",
        3: "Family Bundle",
        4: "Family Plus",
        5: "Business Coverage",
        6: "Senior Coverage",
        7: "High-Value Coverage",
        8: "Flex Bundle",
        9: "Elite Coverage"
    };
    
    const bundleName = bundleNames[bundleNumber] || `Bundle ${bundleNumber}`;
    
    resultContent.innerHTML = `
        <div class="bundle-prediction">Bundle ${bundleNumber}</div>
        <div style="font-size: 1.5em; margin: 15px 0;">${bundleName}</div>
        <div class="bundle-status">✓ Recommendation generated successfully</div>
    `;
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

function displayError(message) {
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    resultContent.innerHTML = `
        <div class="error">❌ ${message}</div>
    `;
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

async function loadBundlesInfo() {
    try {
        const response = await fetch('/api/bundle-info');
        const data = await response.json();
        const bundlesGrid = document.getElementById('bundlesInfo');
        
        bundlesGrid.innerHTML = '';
        
        if (data.bundles) {
            Object.entries(data.bundles).forEach(([id, bundle]) => {
                const card = document.createElement('div');
                card.className = 'bundle-card';
                card.innerHTML = `
                    <div class="bundle-id">Bundle ${id}</div>
                    <h3>${bundle.name}</h3>
                    <p>${bundle.description}</p>
                `;
                bundlesGrid.appendChild(card);
            });
        }
    } catch (error) {
        document.getElementById('bundlesInfo').innerHTML = 
            `<p style="color: red;">Error loading bundle information: ${error.message}</p>`;
    }
}

// Load bundles info on page load if needed
window.addEventListener('load', function() {
    console.log('Insurance Recommender API loaded');
});

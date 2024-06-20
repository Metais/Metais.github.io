document.addEventListener("DOMContentLoaded", function() {
    // Function to populate drug options
    function populateDrugOptions(drugs) {
        const drugSelect = document.getElementById('drug');
        for (const drug in drugs) {
            const option = document.createElement('option');
            option.value = drug;
            option.textContent = drug;
            drugSelect.appendChild(option);
        }
    }

    // Function to update indications based on selected drug
    function updateIndications(drugs) {
        const drugSelect = document.getElementById('drug');
        const indicationSelect = document.getElementById('indication');
        
        drugSelect.addEventListener('change', function() {
            indicationSelect.innerHTML = '<option value="" selected disabled>Select an indication</option>';
            const selectedDrug = drugSelect.value;
            if (selectedDrug) {
                drugs[selectedDrug].forEach(indication => {
                    const option = document.createElement('option');
                    option.value = indication;
                    option.textContent = indication;
                    indicationSelect.appendChild(option);
                });
            }
        });
    }

    // Fetch drug data from JSON file
    fetch('drugs.json')
        .then(response => response.json())
        .then(drugs => {
            populateDrugOptions(drugs);
            updateIndications(drugs);

            // Handle form submission
            document.getElementById('selectionForm').addEventListener('submit', function(event) {
                event.preventDefault();
                const drug = document.getElementById('drug').value;
                const ageGroup = document.getElementById('ageGroup').value;
                const indication = document.getElementById('indication').value;
                sessionStorage.setItem('drug', drug);
                sessionStorage.setItem('ageGroup', ageGroup);
                sessionStorage.setItem('indication', indication);
                window.location.href = 'assessment.html';
            });
        })
        .catch(error => console.error('Error loading drugs:', error));
});
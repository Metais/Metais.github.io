document.addEventListener("DOMContentLoaded", function() {
  // Example drug data
  const drugs = {
      "Aspirin": ["Headache", "Pain", "Fever"],
      "Ibuprofen": ["Inflammation", "Pain", "Fever"]
  };

  const drugSelect = document.getElementById('drug');
  const indicationSelect = document.getElementById('indication');

  // Function to populate drug options
  function populateDrugOptions() {
    for (const drug in drugs) {
        const option = document.createElement('option');
        option.value = drug;
        option.textContent = drug;
        drugSelect.appendChild(option);
    }
  }

  // Initial population of drug options
  populateDrugOptions();

  // Update indications based on selected drug
  drugSelect.addEventListener('change', function() {
      indicationSelect.innerHTML = '';
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

  // Handle form submission
  document.getElementById('selectionForm').addEventListener('submit', function(event) {
      event.preventDefault();
      const drug = drugSelect.value;
      const ageGroup = document.getElementById('ageGroup').value;
      const indication = indicationSelect.value;
      sessionStorage.setItem('drug', drug);
      sessionStorage.setItem('ageGroup', ageGroup);
      sessionStorage.setItem('indication', indication);
      window.location.href = 'assessment.html';
  });

  // Populate assessment form
  if (window.location.pathname.endsWith('assessment.html')) {
      const questions = {
          "common": [
              { "question": "Is the patient currently on any medication?", "type": "text" },
              { "question": "Has the patient experienced any adverse effects?", "type": "text" }
          ],
          "Aspirin": [
              { "question": "Is the patient allergic to NSAIDs?", "type": "text" }
          ],
          "Ibuprofen": [
              { "question": "Does the patient have a history of stomach ulcers?", "type": "text" }
          ]
      };

      const drug = sessionStorage.getItem('drug');
      const questionsContainer = document.getElementById('questionsContainer');

      questions.common.forEach(q => {
          const label = document.createElement('label');
          label.textContent = q.question;
          const input = document.createElement('input');
          input.type = q.type;
          questionsContainer.appendChild(label);
          questionsContainer.appendChild(input);
      });

      if (questions[drug]) {
          questions[drug].forEach(q => {
              const label = document.createElement('label');
              label.textContent = q.question;
              const input = document.createElement('input');
              input.type = q.type;
              questionsContainer.appendChild(label);
              questionsContainer.appendChild(input);
          });
      }

      document.getElementById('assessmentForm').addEventListener('submit', function(event) {
          event.preventDefault();
          // Collect answers and generate summary
          const answers = [];
          questionsContainer.querySelectorAll('input').forEach(input => {
              answers.push(input.value);
          });
          sessionStorage.setItem('answers', JSON.stringify(answers));
          window.location.href = 'summary.html';
      });
  }

  // Generate summary
  if (window.location.pathname.endsWith('summary.html')) {
      const drug = sessionStorage.getItem('drug');
      const ageGroup = sessionStorage.getItem('ageGroup');
      const indication = sessionStorage.getItem('indication');
      const answers = JSON.parse(sessionStorage.getItem('answers'));
      const summaryContainer = document.getElementById('summaryContainer');
      
      const summary = `
          <p>Drug: ${drug}</p>
          <p>Age Group: ${ageGroup}</p>
          <p>Indication: ${indication}</p>
          <p>Answers: ${answers.join(', ')}</p>
      `;
      summaryContainer.innerHTML = summary;
  }
});

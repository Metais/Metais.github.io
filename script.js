document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('questionnaire');
    const summaryDiv = document.getElementById('summary');
    const summaryText = document.getElementById('summaryText');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault();
  
      const formData = new FormData(form);
      const name = formData.get('name');
      const age = formData.get('age');
      const color = formData.get('color');
  
      const summary = `
        <strong>Name:</strong> ${name}<br>
        <strong>Age:</strong> ${age}<br>
        <strong>Favorite Color:</strong> ${color}
      `;
  
      summaryText.innerHTML = summary;
      form.classList.add('hidden');
      summaryDiv.classList.remove('hidden');
    });
  });
  
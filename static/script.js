const submitBtn = document.getElementById('submitBtn');
const promptInput = document.getElementById('promptInput');
const matchesContainer = document.getElementById('matchesContainer');

submitBtn.addEventListener('click', () => {
  const prompt = promptInput.value;
  if (prompt.trim() === '') {
    alert('Please enter a prompt');
    return;
  }

  const requestBody = { prompt: prompt };

  fetch('/find_matches', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestBody)
  })
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Request failed');
      }
    })
    .then(data => {
      const matches = data.matches;
      displayMatches(matches);
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while fetching matches');
    });
});

// function displayMatches(matches) {
//   matchesContainer.innerHTML = '';
//   if (matches.length === 0) {
//     matchesContainer.innerHTML = '<p>No matches found</p>';
//   } else {
//     const matchesList = document.createElement('ul');
//     matches.forEach(match => {
//       const matchItem = document.createElement('li');
//       matchItem.textContent = match;
//       matchesList.appendChild(matchItem);
//     });
//     matchesContainer.appendChild(matchesList);
//   }
// }


function displayMatches(matches) {
  matchesContainer.innerHTML = '';
  if (matches.length === 0) {
    matchesContainer.innerHTML = '<p>No matches found</p>';
  } else {
    const matchesList = document.createElement('ul');
    matches.forEach(match => {
      const matchItem = document.createElement('li');
      const img = document.createElement('img');
      img.src = match;
      img.alt = 'Match Image';
      matchItem.appendChild(img);
      matchesList.appendChild(matchItem);
    });
    matchesContainer.appendChild(matchesList);
  }
}


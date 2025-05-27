/**
 * This is a simplified script that outputs only JSON to avoid
 * TensorFlow warnings and other logging that interferes with parsing
 */

const fs = require('fs');
const path = require('path');

function mockPredict(audioPath) {
  // Extract filename to get some variability
  const filename = path.basename(audioPath);
  const fileSum = [...filename].reduce((sum, char) => sum + char.charCodeAt(0), 0);
  
  // Map to a genre based on simple hashing
  const genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock'];
  const primaryGenreIdx = fileSum % genres.length;
  const primaryGenre = genres[primaryGenreIdx];
  
  // Create probabilities
  const probs = Object.fromEntries(genres.map(g => [g, 0.05]));
  probs[primaryGenre] = 0.6;
  
  // Choose two secondary genres with higher probability
  const secondaryGenres = genres.filter(g => g !== primaryGenre).slice(0, 2);
  secondaryGenres.forEach(g => { probs[g] = 0.15; });
  
  // Return the results
  return {
    genre_probabilities: probs,
    predicted_genre: primaryGenre,
    confidence: 0.6,
    using_mock: true
  };
}

function mockRecommendations(audioPath, topK = 5) {
  // Match the genre with the prediction
  const prediction = mockPredict(audioPath);
  const primaryGenre = prediction.predicted_genre;
  
  // Generate recommendations
  const recommendations = [];
  
  for (let i = 0; i < topK; i++) {
    // 80% chance to match primary genre, 20% for random
    const useGenre = Math.random() < 0.8 
      ? primaryGenre 
      : ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
        [Math.floor(Math.random() * 8)];
    
    recommendations.push({
      id: String(10000 + i),
      title: `${useGenre} Track ${i+1}`,
      genre: useGenre,
      similarity: 0.95 - (i * 0.05)
    });
  }
  
  return recommendations;
}

// This script is meant to be called directly from Node.js
const args = process.argv.slice(2);
if (args.length >= 2) {
  const command = args[0];
  const filePath = args[1];
  
  if (command === 'predict') {
    console.log(JSON.stringify(mockPredict(filePath)));
  } else if (command === 'recommend') {
    const count = args[2] ? parseInt(args[2], 10) : 5;
    console.log(JSON.stringify(mockRecommendations(filePath, count)));
  }
}

module.exports = {
  mockPredict,
  mockRecommendations
};

// Speech functionality for yoga poses
let speechEnabled = false;
let intervalId = null;

function toggleSpeech() {
    speechEnabled = !speechEnabled;
    const button = document.getElementById('toggle-speech');
    button.textContent = `Speech: ${speechEnabled ? 'On' : 'Off'}`;
    
    if (speechEnabled) {
        startSpeechInstructions();
    } else {
        stopSpeechInstructions();
    }
}

function startSpeechInstructions() {
    const instructions = document.querySelector('.pose-description p').textContent;
    const speech = new SpeechSynthesisUtterance(instructions);
    speech.rate = 0.8; // Slightly slower rate for better understanding
    speech.pitch = 1;
    window.speechSynthesis.speak(speech);
}

function stopSpeechInstructions() {
    window.speechSynthesis.cancel();
}

// Add event listener for page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden && speechEnabled) {
        stopSpeechInstructions();
    }
}); 
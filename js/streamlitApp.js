// script.js
document.addEventListener("DOMContentLoaded", function() {
    console.log("Custom JavaScript loaded!");
    
    // Add any custom JavaScript functionality here
    document.querySelectorAll('.skill-chip').forEach(chip => {
        chip.addEventListener('click', function() {
            alert('Skill clicked: ' + this.textContent);
        });
    });
});

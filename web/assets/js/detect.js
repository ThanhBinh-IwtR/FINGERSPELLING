const darkModeButton = document.querySelector('.dark-mode-button');

darkModeButton.addEventListener('click', () => {
  document.body.classList.toggle('dark-mode');
});
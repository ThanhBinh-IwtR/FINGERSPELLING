const button = document.querySelector('.sectionbutton button');

button.addEventListener('click', () => {
  document.body.classList.add('hide-scroll', 'fade-out');
  setTimeout(() => {
    window.location.href = 'detect.html';
  }, 1000);
});
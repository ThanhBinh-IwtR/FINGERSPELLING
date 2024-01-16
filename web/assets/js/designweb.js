//transitions
document.body.classList.add('fade-in');

//darkmode
const themeToggle = document.getElementById('theme');
const body = document.body;

themeToggle.onclick = () => {
  body.classList.toggle('dark-mode');
  themeToggle.classList.toggle('fa-moon');
  themeToggle.classList.toggle('fa-sun');
  if (body.classList.contains('dark-mode')) {
    // Change to white logo when in dark mode
    logo.src = 'assets/img/logo.png';
  } else {
    // Change to dark logo when not in dark mode
    logo.src = 'assets/img/logoD.png';
  }
}

//height of box
window.addEventListener('load', function() {

  var img = document.getElementById('video-img');
  var textarea = document.getElementById('result-textarea');

  var imgHeight = img.clientHeight;
  textarea.style.height = (imgHeight - 45) + 'px'; 

  // Cập nhật chiều cao của <textarea> khi kích thước cửa sổ thay đổi
  window.addEventListener('resize', function() {
      imgHeight = img.clientHeight;
      textarea.style.height = (imgHeight - 45 ) + 'px'; 
  });
});


//popup
const guideButton = document.querySelector('.home-button:nth-child(2)');
const popup = document.querySelector('.popup');
const popupExitButton = document.querySelector('.popup-exit');

guideButton.addEventListener('click', () => {
    popup.style.display = 'block';
});

popupExitButton.addEventListener('click', () => {
    popup.style.display = 'none';
});
  

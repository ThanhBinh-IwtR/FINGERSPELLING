let checkflag = false;

function hideVideoFeed() {
    var videoFeed = document.getElementById('video-img');
    videoFeed.style.display = 'block'; 
    checkflag = false;

    var userUploadedVideo = document.getElementById('user-uploaded-video');
        userUploadedVideo.pause();
        userUploadedVideo.style.display = 'none';
}

function pollForResults() {
    setInterval(function() {
        if (checkflag == false) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/get_latest_result', true);
    
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    var resultTextarea = document.getElementById('result-textarea');
                    var responseData = JSON.parse(xhr.responseText);
                    var result = responseData.result;
                    resultTextarea.value = result;
                }
            };
    
            xhr.send();
        }   
    }, 2000);  //update result every 2 seconds
}
pollForResults();


//clear result
document.getElementById('clear-button').addEventListener('click', function() {
    fetch('/clear_all_res', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.result === 'success') {
            document.getElementById('result-textarea').value = '';
        }
    });
});

//import video

var uploadPath = window.location.origin + '/assets/upload/';
document.getElementById('upload_button').addEventListener('click', function() {
    var file = document.getElementById('file_input').files[0];
    var formData = new FormData();
    var url = URL.createObjectURL(file);
    var userUploadedVideo = document.getElementById('user-uploaded-video');
        userUploadedVideo.src = url;
        userUploadedVideo.style.display = 'block';
        checkflag = true;
    var videoFeed = document.getElementById('video-img');
        videoFeed.style.display = 'none'; 

    // var videofeed = document.getElementById('livedetect');
    //     videofeed.style.display = 'block';
    formData.append('file', file);
    fetch('/upload', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.result === 'success') {
                var filename = data.filename;
                
                

                var import_result = document.getElementById('result-textarea');
                import_result.value += data.data.join(''); 

               
                
                

            }
        });
});

function updateStats() {
    fetch('/get_FPS_PC')
        .then(response => response.json())
        .then(data => {
            document.getElementById('fps').textContent = 'FPS: ' + data.fps;
            document.getElementById('pc').textContent = 'PC: ' + data.pc;
        });
}

setInterval(updateStats, 1000);
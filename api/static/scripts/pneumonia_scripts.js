var selectedImg;


const loadImg = () => {
    //var preview = document.querySelector('img');
    var file = document.getElementById("selectedImage").files[0];
    var reader = new FileReader();
    reader.addEventListener("load", function () {
        $('#img').attr('width','64').attr('height','64');
        $("#img").attr('src','static/25.gif');
        $('#txtError').text('');

        selectedImg = reader.result;

        axios.post('http://localhost:5000/api/pneumonia/predict', {
            data: selectedImg
        })
        .then((response) => {
            confidence = response.data.confidence;
            bytestring = response.data.image;
            if(bytestring == "-1")
            {
                $('#img').attr('width','400').attr('height','400');
                $("#img").attr('src',selectedImg);
                $('#txtError').text('No instances detected.');
            }
            else
            {
                image = bytestring.split('\'')[1];
                $('#img').attr('width','400').attr('height','400');
                $("#img").attr('src' , 'data:image/jpeg;base64,'+image);
                console.log(confidence);
                $('#txtError').text("Confidence : %" + confidence);
            }
            
        })
        .catch((err) => {
            console.log(err);
        });
    }, false);
    if (file) {
        reader.readAsDataURL(file);
    }
}
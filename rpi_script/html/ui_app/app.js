var send_to_server = function(val) {
    var data = new FormData();
    data.append('belt', val);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://localhost:5000/run', true);
    xhr.onload = function() {
        // do something to response
        console.log(this.responseText);
    };
    xhr.send(data);
}

window.addEventListener("load", function() {
    console.log("Hello World!");
    send_to_server(106);
});
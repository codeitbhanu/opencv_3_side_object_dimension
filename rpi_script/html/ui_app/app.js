var myFunction = function() {
    console.log("Sending Request to Server...")
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log(this.responseText)
            try {
                data = JSON.parse(this.responseText)
                    // console.log("color: " + data.color.front.code)

                var dimensions = data.dimensions;
                var height = dimensions.h;
                var length = dimensions.l;
                var width = dimensions.w;

                var allColor = data.color;
                var frontColor = allColor.front.code;
                var rearColor = allColor.rear.code;

                document.getElementById("colWidth").innerHTML = width + 'mm';
                document.getElementById("colHeight").innerHTML = height + 'mm';
                document.getElementById("colLength").innerHTML = length + 'mm';
                document.getElementById("colFrontColor").bgColor = frontColor;
                document.getElementById("colFrontColor").innerHTML = frontColor;
                document.getElementById("colRearColor").innerHTML = rearColor;
                document.getElementById("colRearColor").bgColor = rearColor;

            } catch (err) {
                console.log('error in returned data: ' + err.message)
            }
        }
    };
    xhttp.open("GET", "http://192.168.225.106:5000/run", true);
    xhttp.send();
}

var myJQueryFunction = function() {
    console.log("Sending Request to Server...")
    $.ajax({
        url: "http://192.168.225.106:5000/run",
        crossDomain: true,
        beforeSend: function(client) {
            //Authorization: "Basic " + btoa(username + ":" + password);
            client.setRequestHeader('Authorization');
            client.setRequestHeader('Access-Control-Allow-Origin','*');
        },
        success: function(result){
        try {
                data = JSON.parse(result)
                // data = JSON.parse(this.responseText)
                    // console.log("color: " + data.color.front.code)

                var dimensions = data.dimensions;
                var height = dimensions.h;
                var length = dimensions.l;
                var width = dimensions.w;

                var allColor = data.color;
                var frontColor = allColor.front.code;
                var rearColor = allColor.rear.code;

                document.getElementById("colWidth").innerHTML = width + 'mm';
                document.getElementById("colHeight").innerHTML = height + 'mm';
                document.getElementById("colLength").innerHTML = length + 'mm';
                document.getElementById("colFrontColor").bgColor = frontColor;
                document.getElementById("colFrontColor").innerHTML = frontColor;
                document.getElementById("colRearColor").innerHTML = rearColor;
                document.getElementById("colRearColor").bgColor = rearColor;

            } catch (err) {
                console.log('error in returned data: ' + err.message)
            }
        }
    });
}



$(document).ready(function(){
    // $("button").click(function(){
    //     $.ajax({url: "demo_test.txt", success: function(result){
    //         $("#div1").html(result);
    //     }});
    // });
});

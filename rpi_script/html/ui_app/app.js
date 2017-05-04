$(document).ready(function() {
    // process the form
    $('form').submit(function(event) {
        // get the form data
        // there are many ways to get this data using jQuery (you can use the class or id also)
        var formData = {'belt':'106'};

        // process the form
        $.ajax({
            type        : 'GET', // define the type of HTTP verb we want to use (POST for our form)
            url         : 'http://localhost:5000/run?belt=106', // the url where we want to POST
            //data        : formData, // our data object
            dataType    : 'json', // what type of data do we expect back from the server
                        encode          : true
        })
            // using the done promise callback
            .done(function(data) {

                // log data to the console so we can see
                console.log(data); 

                // here we will handle errors and validation messages
            });

        // stop the form from submitting the normal way and refreshing the page
        event.preventDefault();
        
        var responseText =  {"color":{"front":{"code":"#16120d","r":22,"b":13,"g":18},
                                   "rear": {"code":"#16120d","r":22,"b":13,"g":18}
                                  },
                          "dimensions":{"h":23.31,"l":40.0,"w":13.71}
                          };    

    var dimensions = responseText.dimensions;
    var height = dimensions.h;
    var length = dimensions.l;
    var width = dimensions.w;
    
    var allColor = responseText.color;
    var frontColor = allColor.front.code;
    var rearColor = allColor.rear.code;
    
    document.getElementById("colWidth").innerHTML = width;
    document.getElementById("colHeight").innerHTML = height;
    document.getElementById("colLength").innerHTML = length;
    document.getElementById("colFrontColor").bgColor = frontColor;
    document.getElementById("colRearColor").bgColor = rearColor;
    
    });

});

var send_to_server = function(val) {
}

//window.addEventListener("load", function() {
  //  console.log("Hello World!");
//    send_to_server(106);
//});

function myFunction()
{
    
    
    
    
    
//    var postData = "text";
//      $.ajax({
//            type: "post",
//            url: "http://192.168.225.106:5000/run?belt=106",
//            data: "106",
//            contentType: "application/x-www-form-urlencoded",
//            success: function(responseData, textStatus, jqXHR) {
//                alert("data saved")
//            },
//            error: function(jqXHR, textStatus, errorThrown) {
//                console.log(errorThrown);
//            }
//        })
    
    
//    var data = new FormData();
//    data.append('belt', '106');
//
//    var xhr = new XMLHttpRequest();
//    xhr.open('POST', 'http://192.168.225.106:5000/run', true);
//    xhr.onload = function() {
//    
//    };
//    xhr.send(data);
    
    var responseText =  {"color":{"front":{"code":"#16120d","r":22,"b":13,"g":18},
                                   "rear": {"code":"#16120d","r":22,"b":13,"g":18}
                                  },
                          "dimensions":{"h":23.31,"l":40.0,"w":13.71}
                          };    

    var dimensions = responseText.dimensions;
    var height = dimensions.h;
    var length = dimensions.l;
    var width = dimensions.w;
    
    var allColor = responseText.color;
    var frontColor = allColor.front.code;
    var rearColor = allColor.rear.code;
    
    document.getElementById("colWidth").innerHTML = width;
    document.getElementById("colHeight").innerHTML = height;
    document.getElementById("colLength").innerHTML = length;
    document.getElementById("colFrontColor").bgColor = frontColor;
    document.getElementById("colRearColor").bgColor = rearColor;
    
    //var obj = JSON.parse(responseText);    

    //alert('Test');
}
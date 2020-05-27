function getUrlParameter() {
    var url = window.location.href;
    var urlSplit = url.split('/');
    var urlLastSegment = urlSplit.pop() || urlSplit.pop();
    return urlLastSegment
}
 
$(document).ready(function() {
    // Get URL parameter
    dynamicContent = getUrlParameter()
    console.log(dynamicContent)
    if (dynamicContent == 'knowledge-graph-system') {
        $('#knowledge-graph-system').show();
    }
    else if (dynamicContent == 'paper-recommender-system') {
        $('#paper-recommender-system').show();
        // pagination
        $('#table').DataTable( {
        "pagingType": "full_numbers"
        } );
    }
    else if (dynamicContent == 'q-a-system') {
        $('#q-a-system').show();
        // pagination
        $('#table').DataTable( {
        "pagingType": "full_numbers"
        } );
    }
    else if (dynamicContent == 'about-us') {
        $('#about-us').show();
    }
    else if (dynamicContent == 'search') {
        system_split = window.location.pathname.split("/");
        system = system_split[system_split.length-2]
        console.log("printing system")
        console.log(system)
        if (system == 'paper-recommender-system') {
            $('#paper-recommender-system').show();
            // pagination
            $('#table').DataTable( {
            "pagingType": "full_numbers"
            } );
        }
        else if (system == 'q-a-system') {
            $('#q-a-system').show();
            // pagination
            $('#table').DataTable( {
            "pagingType": "full_numbers"
            } );
        }
        else {
            $('#homepage').show();
        }
    }
    else {
        $('#homepage').show();
    }
});

window.onload = function() {
    // Setup paper-recommender-system slider
    var slider = document.getElementById("paper-recommender-system-search-bar-dropdown-menu-slider");
    var title_weight = document.getElementById("paper-recommender-system-search-bar-dropdown-menu-slider-title-weight");
    var abstract_weight = document.getElementById("paper-recommender-system-search-bar-dropdown-menu-slider-abstract-weight");
    title_weight.innerHTML = slider.value;
    abstract_weight.innerHTML = 100 - slider.value;

    slider.oninput = function() {
        title_weight.innerHTML = this.value;
        abstract_weight.innerHTML = 100 - this.value;
    }

  }

function myFunction(count) {
    var dots = document.getElementById("dots" + count);
    var moreText = document.getElementById("more" + count);
    var btnText = document.getElementById("myBtn" + count);
    if (dots.style.display === "none") {
        dots.style.display = "inline";
        btnText.innerHTML = "Read more";
        moreText.style.display = "none";
    } else {
        dots.style.display = "none";
        btnText.innerHTML = "Read less";
        moreText.style.display = "inline";
    }
}
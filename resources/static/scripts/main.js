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
    else if (dynamicContent == 'paper-recommender-system' || dynamicContent == 'search') {
        $('#paper-recommender-system').show();
    }
    else if (dynamicContent == 'q-a-system') {
        $('#q-a-system').show();
    }
    else if (dynamicContent == 'about-us') {
        $('#about-us').show();
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
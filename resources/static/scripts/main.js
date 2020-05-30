// d3 js force directed graph
function draw_knowledge_graph() {

    var width = 800;
    var height = 600;

    var colors = d3.scaleOrdinal(d3.schemeCategory10);

    var svg = d3.select("#graph"),
        width = +svg.attr("width"),
        height = +svg.attr("height"),
        node,
        link;

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function (d) {return d.id;}).distance(100).strength(1))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(width / 2, height / 2));

    update(graph.links, graph.nodes)

    function update(links, nodes) {
        link = svg.selectAll(".link")
            .data(links)
            .enter()
            .append("line")
            .attr("class", "link")

        edgepaths = svg.selectAll(".edgepath")
            .data(links)
            .enter()
            .append('path')
            .attrs({
                'class': 'edgepath',
                'fill-opacity': 0,
                'stroke-opacity': 0,
                'id': function (d, i) {return 'edgepath' + i}
            })
            .style("pointer-events", "none");

        edgelabels = svg.selectAll(".edgelabel")
            .data(links)
            .enter()
            .append('text')
            .style("pointer-events", "none")
            .attrs({
                'class': 'edgelabel',
                'id': function (d, i) {return 'edgelabel' + i},
                'font-size': 10,
                'fill': '#aaa'
            });

        edgelabels.append('textPath')
            .attr('xlink:href', function (d, i) {return '#edgepath' + i})
            .style("text-anchor", "middle")
            .style("pointer-events", "none")
            .attr("startOffset", "50%")
            .text(function (d) {return d.name_of_relation}
            );

        node = svg.selectAll(".node")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node")
            .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
            );

        // Set up node color - type_of_entity
        node.append("circle")
            .attr("r", 5)
            .style("fill", function (d, i) {return colors(d.type_of_entity);})

        // Set up node name - name_of_entity
        node.append("text")
            .attr("dy", -3)
            .text(function (d) {return d.name_of_entity;});

        simulation
            .nodes(nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(links);

    }

    function ticked() {

        link.attr("x1", function (d) {return d.source.x;})
            .attr("y1", function (d) {return d.source.y;})
            .attr("x2", function (d) {return d.target.x;})
            .attr("y2", function (d) {return d.target.y;});

        node.attr("transform", function (d) {return "translate(" + d.x + ", " + d.y + ")";});

        edgepaths.attr('d', function (d) {
            return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
        });

        edgelabels.attr('transform', function (d) {
            if (d.target.x < d.source.x) {
                var bbox = this.getBBox();

                rx = bbox.x + bbox.width / 2;
                ry = bbox.y + bbox.height / 2;
                return 'rotate(180 ' + rx + ' ' + ry + ')';
            }
            else {
                return 'rotate(0)';
            }
        });

    }

    function dragstarted(d) {
        if (!d3.event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    }

}

// d3 js legend
function draw_knowledge_graph_legend() {

    var svg_legend = d3.select("#legend");
    var color = d3.scaleOrdinal(d3.schemeCategory10);

    svg_legend.selectAll("mydots")
        .data(keys)
        .enter()
        .append("circle")
        .attr("cx", 100)
        .attr("cy", function(d,i){ return 100 + i*25 })
        .attr("r", 7)
        .style("fill", function(d){ return color(d)});

    svg_legend.selectAll("mylabels")
        .data(keys)
        .enter()
        .append("text")
        .attr("x", 120)
        .attr("y", function(d,i){ return 100 + i*25 })
        .style("fill", function(d){ return color(d) })
        .text(function(d){ return d })
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")
}

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
        draw_knowledge_graph();
        draw_knowledge_graph_legend();
    }
    else if (dynamicContent == 'paper-recommender-system') {
        $('#paper-recommender-system').show();
        // pagination
    }
    else if (dynamicContent == 'q-a-system') {
        $('#q-a-system').show();
    }
    else if (dynamicContent == 'about-us') {
        $('#about-us').show();
    }
    else if (dynamicContent == 'search') {
        system_split = window.location.pathname.split("/");
        system = system_split[system_split.length-2]
        if (system == 'paper-recommender-system') {
            $('#paper-recommender-system').show();
        }
        else if (system == 'q-a-system') {
            $('#q-a-system').show();
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

$(document).ready(function(){
    $('table').DataTable({
        "pageLength": 10,
        autoWidth: false,
        "columns": [
            { "width": "10%" },
            { "width": "30%" },
            { "width": "30%" },
            { "width": "10%" },
            { "width": "10%" },
            { "width": "10%" }
          ],
        searching: true
    });
});

$.extend( $.fn.dataTable.defaults, {
    searching: false,
    ordering:  false
} );

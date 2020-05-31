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
            .on("click", function(d) {
                document.getElementById('paper-title').innerHTML = d.name_of_paper;
                document.getElementById('name-of-relation').innerHTML = 'Relation = ' + d.name_of_relation;
                document.getElementById('link-information').style.display = "block";
                document.getElementById('url-of-paper').href = d.url_of_paper;
                var cord_uid_specific_graph = {
                    "nodes": [],
                    "links": []
                };
                nodes = graph.nodes;
                var i;
                for (i=0 ; i<nodes.length; i++) {
                    console.log(nodes[i].cord_uid); 
                    console.log(d.cord_uid);
                    if(nodes[i].cord_uid === d.cord_uid) {
                        console.log("Matching Edge"); 
                        cord_uid_specific_graph.nodes.push(nodes[i]);
                    }
                }
                links = graph.links;
                for (i=0; i<links.length; i++) {
                    console.log(links[i]);
                    if(links[i].cord_uid === d.cord_uid) {
                        console.log("Matching link");
                        cord_uid_specific_graph.links.push(links[i]);
                    }
                }
                console.log(cord_uid_specific_graph)
                draw_knowledge_graph_paper_specific(cord_uid_specific_graph)
                   
            })
            .on("mouseover", function(d) {
                d3.select(this).style("stroke", "red");
            })                  
            .on("mouseout", function(d) {
                d3.select(this).style("stroke", "#999");
            });

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

// d3 js force directed graph
function draw_knowledge_graph_paper_specific(cord_uid_specific_graph) {

    var width = 800;
    var height = 600;

    var colors = d3.scaleOrdinal(d3.schemeCategory10);


    $("#graph_specific").empty();

    var svg = d3.select("#graph_specific"),
        width = +svg.attr("width"),
        height = +svg.attr("height"),
        node,
        link_specific;

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function (d) {return d.id;}).distance(100).strength(1))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(width / 2, height / 2));

    update(cord_uid_specific_graph.links, cord_uid_specific_graph.nodes)

    function update(links, nodes) {
        link_specific = svg.selectAll(".link_specific")
            .data(links)
            .enter()
            .append("line")
            .attr("class", "link_specific")
            .on("mouseover", function(d) {
                d3.select(this).style("stroke", "red");
            })                  
            .on("mouseout", function(d) {
                d3.select(this).style("stroke", "#999");
            });

        edgepaths_specific = svg.selectAll(".edgepath_specific")
            .data(links)
            .enter()
            .append('path')
            .attrs({
                'class': 'edgepath_specific',
                'fill-opacity': 0,
                'stroke-opacity': 0,
                'id': function (d, i) {return 'edgepath_specific' + i}
            })
            .style("pointer-events", "none");

        edgelabels_specific = svg.selectAll(".edgelabel_specific")
            .data(links)
            .enter()
            .append('text')
            .style("pointer-events", "none")
            .attrs({
                'class': 'edgelabel_specific',
                'id': function (d, i) {return 'edgelabel_specific' + i},
                'font-size': 10,
                'fill': '#aaa'
            });

            edgelabels_specific.append('textPath')
            .attr('xlink:href', function (d, i) {return '#edgepath_specific' + i})
            .style("text-anchor", "middle")
            .style("pointer-events", "none")
            .attr("startOffset", "50%")
            .text(function (d) {return d.name_of_relation}
            );

        node_specific = svg.selectAll(".node_specific")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node_specific")
            .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
            );

        // Set up node color - type_of_entity
        node_specific.append("circle")
            .attr("r", 5)
            .style("fill", function (d, i) {return colors(d.type_of_entity);})

        // Set up node name - name_of_entity
        node_specific.append("text")
            .attr("dy", -3)
            .text(function (d) {return d.name_of_entity;});

        simulation
            .nodes(nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(links);

    }

    function ticked() {

        link_specific.attr("x1", function (d) {return d.source.x;})
            .attr("y1", function (d) {return d.source.y;})
            .attr("x2", function (d) {return d.target.x;})
            .attr("y2", function (d) {return d.target.y;});

            node_specific.attr("transform", function (d) {return "translate(" + d.x + ", " + d.y + ")";});

        edgepaths_specific.attr('d', function (d) {
            return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
        });

        edgelabels_specific.attr('transform', function (d) {
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

// d3 js legend
function draw_knowledge_graph_legend_paper_specific() {

    var svg_legend = d3.select("#legend_specific");
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

function filter(cord_uid, graph) {
    cord_uid_specific_graph = {
        "nodes": [],
        "links": []
    };
    nodes = graph.nodes;
    console.log(nodes);
    for (node in nodes) {
        if(node.cord_uid === cord_uid) {
            console.log("comig");
            cord_uid_specific_graph.nodes.push(node.cord_uid);
        }
    }
    links = graph.links;
    for (link in links) {
        if(link.cord_uid === cord_uid) {
            cord_uid_specific_graph.links.push(link.cord_uid);
        }
    }
    return cord_uid_specific_graph
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
            $(document).ready(function(){
            $('#table_p_r').DataTable({
                "pageLength": 10,
                searching: true,
                autoWidth: false,
                "columns": [
                    { "width": "5%" },
                    { "width": "5%" },
                    { "width": "10%" },
                    { "width": "50%" },
                    { "width": "10%" },
                    { "width": "10%" },
                    { "width": "10%" }
                  ]
                });
            });
        }
        else if (system == 'q-a-system') {
            $('#q-a-system').show();
            $(document).ready(function(){
            $('#table_q_a').DataTable({
                "pageLength": 10,
                searching: true,
                autoWidth: false,
                "columns": [
                    { "width": "5%" },
                    { "width": "5%" },
                    { "width": "10%" },
                    { "width": "20%" },
                    { "width": "30%" },
                    { "width": "10%" },
                    { "width": "10%" },
                    { "width": "10%" }
                  ]
                });
            });
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


$.extend( $.fn.dataTable.defaults, {
    searching: false,
    ordering:  false
} );

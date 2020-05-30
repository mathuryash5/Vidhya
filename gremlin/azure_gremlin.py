import logging

from gremlin_python.driver import client


class MicrosoftAzureCosmosDBGremlinAPI:

    client = None

    _gremlin_traversals = {
        "GET ALL VERTICES": "g.V()",
        "GET ALL EDGES": "g.E()"
    }

    keys = None
    graph = None

    @staticmethod
    def execute_traversals(query, client):
        callback = MicrosoftAzureCosmosDBGremlinAPI.client.submitAsync(MicrosoftAzureCosmosDBGremlinAPI._gremlin_traversals[query])
        for result in callback.result():
            return result

    @staticmethod
    def unpack_knowledge_graph(graph, legend, vertices, edges):
        legend_set = set()
        for vertex in vertices:
            id = vertex['id']
            name_of_entity = vertex['properties']['NameOfEntity'][0]['value']
            type_of_entity = vertex['properties']['TypeOfEntity'][0]['value'].upper()
            legend_set.add(type_of_entity)
            vertex_dict = {'id':id, 'name_of_entity': name_of_entity, 'type_of_entity': type_of_entity}
            graph['nodes'].append(vertex_dict)
        for edge in edges:
            name_of_relation = edge['label']
            source = edge['inV']
            target = edge['outV']
            name_of_paper = edge['properties']['NameOfPaper']
            url_of_paper = edge['properties']['UrlOfPaper']
            edge_dict = {'name_of_relation': name_of_relation, 'source': source, 'target': target, 'name_of_paper': name_of_paper, 'url_of_paper': url_of_paper}
            graph['links'].append(edge_dict)
        legend = list(legend_set)
        return graph, legend

    @staticmethod
    def get_knowledge_graph():
        graph = {
            "nodes": [],
            "links":[]
        }
        legend = []
        vertices = MicrosoftAzureCosmosDBGremlinAPI.execute_traversals("GET ALL VERTICES", MicrosoftAzureCosmosDBGremlinAPI.client)
        edges = MicrosoftAzureCosmosDBGremlinAPI.execute_traversals("GET ALL EDGES", MicrosoftAzureCosmosDBGremlinAPI.client)
        graph, legend = MicrosoftAzureCosmosDBGremlinAPI.unpack_knowledge_graph(graph, legend, vertices, edges)
        return graph, legend

    @staticmethod
    def setup_gremlin():
        logging.debug("Connecting to Microsoft Azure Gremlin API")
        try:
            MicrosoftAzureCosmosDBGremlinAPI.client = client.Client('wss://vidhya-gremlin.gremlin.cosmosdb.azure.com:443/', 'g',
                                   username="/dbs/sample-knowledge-graph-database/colls/sample-knowledge-graph-database",
                                   password="DWgufnMRQXA4mzI4is8EgoN15NGb8lffMAVCPrqmQFnQS3L3g6Twdzk2gnEMwe2XPgXY1fz16P3EyThOXNBLmQ=="
                                   )

            logging.debug("Connecting to Microsoft Azure Gremlin API completed... ")
            logging.debug("Collecting data from Microsoft Azure Gremlin API...")
            MicrosoftAzureCosmosDBGremlinAPI.graph, MicrosoftAzureCosmosDBGremlinAPI.keys = MicrosoftAzureCosmosDBGremlinAPI.get_knowledge_graph()
            logging.debug("Collecting data from Microsoft Azure Gremlin API completed...")
        except Exception as e:
            logging.debug('There was an exception: {0}'.format(e))

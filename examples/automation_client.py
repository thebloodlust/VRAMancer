"""
Exemple de client Python pour l’API d’automatisation avancée (REST & GraphQL)
"""
import requests

# REST : créer un job
def create_job(name):
    r = requests.post("http://localhost:5002/api/jobs", json={"name": name})
    print("Job créé:", r.json())

# REST : lister les jobs
def list_jobs():
    r = requests.get("http://localhost:5002/api/jobs")
    print("Jobs:", r.json())

# GraphQL : lister les jobs
def graphql_jobs():
    query = '{ jobs { id name status } }'
    r = requests.post("http://localhost:5002/graphql", json={"query": query})
    print("GraphQL jobs:", r.json())

if __name__ == "__main__":
    create_job("Test pipeline")
    list_jobs()
    graphql_jobs()

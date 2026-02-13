import requests
import json

def check_status():
    try:
        # Check recommendations
        r_recs = requests.get("http://localhost:8000/api/recommendations")
        recs = r_recs.json()
        print(f"Recommendations count: {len(recs)}")
        print(json.dumps(recs, indent=2))

        # Check nodes to see risk scores (closest proxy we have to simulation state)
        r_nodes = requests.get("http://localhost:8000/api/nodes")
        nodes = r_nodes.json()
        if nodes:
            avg_risk = sum(n.get('risk_score', 0) for n in nodes) / len(nodes)
            print(f"Average Node Risk: {avg_risk}")
            print(f"Sample Node Risk: {nodes[0].get('risk_score')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()

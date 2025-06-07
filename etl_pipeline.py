from etl.db import get_collection


def run_pipeline():
    collection = get_collection("example_collection")

    # Add some data to the collection
    collection.upsert(
        documents=["Hello, world!", "This is a test document."],
        ids=["doc1", "doc2"],
    )

    # Query the collection
    results = collection.query(
        query_texts=["Hello"],
        include=['distances', 'documents', 'metadatas']
    )
    print("Query results:", results)


if __name__ == "__main__":
   run_pipeline()

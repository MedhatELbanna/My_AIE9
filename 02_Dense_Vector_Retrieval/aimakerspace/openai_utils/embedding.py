from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import openai
from typing import List
import os
import asyncio


class EmbeddingModel:
    def __init__(self, embeddings_model_name: str = "text-embedding-3-small", batch_size: int = 1024):
        # Try to load .env from current directory and likely locations
        load_dotenv()
        # Try loading from the directory containing this file (going up to project root)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_file_dir, '../../.env'),  # 02_Dense_Vector_Retrieval/.env
            os.path.join(current_file_dir, '../../../.env'),  # My_AIE9/.env
            '.env'  # Current working directory
        ]
        for env_path in possible_paths:
            env_path = os.path.abspath(env_path)
            if os.path.exists(env_path):
                load_dotenv(env_path)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key."
            )
        self.async_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embeddings_model_name = embeddings_model_name
        self.batch_size = batch_size

    async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        batches = [list_of_text[i:i + self.batch_size] for i in range(0, len(list_of_text), self.batch_size)]
        
        async def process_batch(batch):
            embedding_response = await self.async_client.embeddings.create(
                input=batch, model=self.embeddings_model_name
            )
            return [embeddings.embedding for embeddings in embedding_response.data]
        
        # Use asyncio.gather to process all batches concurrently
        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten the results
        return [embedding for batch_result in results for embedding in batch_result]

    async def async_get_embedding(self, text: str) -> List[float]:
        embedding = await self.async_client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )

        return embedding.data[0].embedding

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        embedding_response = self.client.embeddings.create(
            input=list_of_text, model=self.embeddings_model_name
        )

        return [embeddings.embedding for embeddings in embedding_response.data]

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.client.embeddings.create(
            input=text, model=self.embeddings_model_name
        )

        return embedding.data[0].embedding


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    print(asyncio.run(embedding_model.async_get_embedding("Hello, world!")))
    print(
        asyncio.run(
            embedding_model.async_get_embeddings(["Hello, world!", "Goodbye, world!"])
        )
    )

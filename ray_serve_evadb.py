
from ray import serve
import evadb
from starlette.requests import Request


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class EvaDB:
    def __init__(self):
        self.cursor = evadb.connect().cursor()

    def execute(self, query: str) -> str:
        return self.cursor.query(query).df()

    async def __call__(self, http_request: Request) -> str:
        query: str = await http_request.json()
        return self.execute(query)


eva_app = EvaDB.bind()

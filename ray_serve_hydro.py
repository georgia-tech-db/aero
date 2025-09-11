from ray import serve

from starlette.requests import Request

from ray import serve

from eva.configuration.configuration_manager import ConfigurationManager
from eva.executor.execution_context import Context
from eva.server.command_handler import execute_query_fetch_all
from eva.utils.stats import Timer


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Hydro:
    def __init__(self):
        self.config = ConfigurationManager()
        self.config.update_value(
            "experimental", "logical_filter_to_physical_rule_workers", 1
        )
        self.config.update_value(
            "experimental", "logical_filter_to_physical_rule_gpus", 1
        )
        self.config.update_value(
            "experimental", "logical_get_to_sequential_scan_workers", 1
        )
        self.config.update_value(
            "experimental", "logical_get_to_sequential_scan_gpus", 1
        )

    def execute(self, query: str) -> str:
        batch = execute_query_fetch_all(query)
        return batch.frames

    async def __call__(self, http_request: Request) -> str:
        query: str = await http_request.json()
        return self.execute(query)


eva_app = Hydro.bind()

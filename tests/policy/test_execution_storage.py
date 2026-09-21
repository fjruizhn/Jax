from policy.execution_control.storage import InMemoryExecutionStore, MariaDBExecutionStore
def test_stores_exist(): assert InMemoryExecutionStore() and MariaDBExecutionStore

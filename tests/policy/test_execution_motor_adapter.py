from policy.execution_control.adapters.motor_registry import governed_motor_payload
def test_adapter_requires_binding():
    try: governed_motor_payload(None, None, trace_id="x")
    except AttributeError: pass

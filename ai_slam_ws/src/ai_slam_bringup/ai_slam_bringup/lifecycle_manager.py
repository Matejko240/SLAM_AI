import time

import rclpy
from rclpy.node import Node

from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition
from rclpy.exceptions import ParameterAlreadyDeclaredException

class LifecycleManager(Node):
    def __init__(self):
        super().__init__("lifecycle_manager")
        try:
            self.declare_parameter("nodes", [""])  # wymusza STRING_ARRAY
        except ParameterAlreadyDeclaredException:
            pass  # jeśli launch już „zadeklarował” parametr

        nodes = list(self.get_parameter("nodes").value)
        nodes = [n for n in nodes if n]  # usuń pusty string
        self.nodes = nodes

        self.timer = self.create_timer(1.0, self.tick)
        self.done = set()

    def _srv_name(self, node, suffix):
        if node.startswith("/"):
            return f"{node}{suffix}"
        return f"/{node}{suffix}"

    def _wait_service(self, client, timeout_sec):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if client.wait_for_service(timeout_sec=0.2):
                return True
        return False

    def _get_state(self, node_name):
        cli = self.create_client(GetState, self._srv_name(node_name, "/get_state"))
        if not self._wait_service(cli, 1.5):
            self.destroy_client(cli)
            return None
        req = GetState.Request()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.5)
        self.destroy_client(cli)
        if fut.done() and fut.result() is not None:
            return int(fut.result().current_state.id)
        return None

    def _change_state(self, node_name, transition_id):
        cli = self.create_client(ChangeState, self._srv_name(node_name, "/change_state"))
        if not self._wait_service(cli, 1.5):
            self.destroy_client(cli)
            return False
        req = ChangeState.Request()
        req.transition.id = int(transition_id)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        self.destroy_client(cli)
        if fut.done() and fut.result() is not None:
            return bool(fut.result().success)
        return False

    def tick(self):
        for n in self.nodes:
            if n in self.done:
                continue

            st = self._get_state(n)
            if st is None:
                self.done.add(n)
                continue

            if st == 1:
                ok = self._change_state(n, Transition.TRANSITION_CONFIGURE)
                if ok:
                    return
                self.done.add(n)
                continue

            if st == 2:
                ok = self._change_state(n, Transition.TRANSITION_ACTIVATE)
                if ok:
                    return
                self.done.add(n)
                continue

            if st == 3:
                self.done.add(n)
                continue

            self.done.add(n)

        if len(self.done) == len(self.nodes):
            self.timer.cancel()


def main():
    rclpy.init()
    node = LifecycleManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

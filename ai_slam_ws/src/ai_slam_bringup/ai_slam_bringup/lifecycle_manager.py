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
        self.declare_parameter("service_wait_timeout_sec", 1.5)
        self.declare_parameter("service_poll_timeout_sec", 0.2)
        self.declare_parameter("get_state_timeout_factor", 1.0)
        self.declare_parameter("change_state_timeout_factor", 1.33)

        nodes = list(self.get_parameter("nodes").value)
        nodes = [n for n in nodes if n]  # usuń pusty string
        self.nodes = nodes
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.service_poll_timeout_sec = max(0.05, float(self.get_parameter("service_poll_timeout_sec").value))
        self.get_state_timeout_factor = max(0.25, float(self.get_parameter("get_state_timeout_factor").value))
        self.change_state_timeout_factor = max(0.25, float(self.get_parameter("change_state_timeout_factor").value))
        self.get_state_call_timeout_sec = self.service_wait_timeout_sec * self.get_state_timeout_factor
        self.change_state_call_timeout_sec = self.service_wait_timeout_sec * self.change_state_timeout_factor

        self.timer = self.create_timer(1.0, self.tick)
        self.done = set()
        self.unavailable_counts = {}

    def _srv_name(self, node, suffix):
        if node.startswith("/"):
            return f"{node}{suffix}"
        return f"/{node}{suffix}"

    def _wait_service(self, client, timeout_sec):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if client.wait_for_service(timeout_sec=self.service_poll_timeout_sec):
                return True
        return False

    def _get_state(self, node_name):
        cli = self.create_client(GetState, self._srv_name(node_name, "/get_state"))
        if not self._wait_service(cli, self.service_wait_timeout_sec):
            self.destroy_client(cli)
            return None
        req = GetState.Request()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.get_state_call_timeout_sec)
        self.destroy_client(cli)
        if fut.done() and fut.result() is not None:
            return int(fut.result().current_state.id)
        return None

    def _change_state(self, node_name, transition_id):
        cli = self.create_client(ChangeState, self._srv_name(node_name, "/change_state"))
        if not self._wait_service(cli, self.service_wait_timeout_sec):
            self.destroy_client(cli)
            return False
        req = ChangeState.Request()
        req.transition.id = int(transition_id)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.change_state_call_timeout_sec)
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
                cnt = self.unavailable_counts.get(n, 0) + 1
                self.unavailable_counts[n] = cnt
                if cnt % 10 == 0:
                    self.get_logger().warn(
                        f"Lifecycle service for '{n}' unavailable for {cnt}s, retrying..."
                    )
                continue

            self.unavailable_counts[n] = 0

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
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

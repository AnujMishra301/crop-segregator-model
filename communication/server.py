"""
Mission Controller Socket Server Module
Runs on the flight control companion computer / ground station side.
Ingests AI detection streams, checks protocol health, evaluates multi-condition safety gates,
and logs SIMULATED spray trigger actions (hardware actuation disabled by default).
"""

import os
import sys
import time
import json
import socket
import threading
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from communication.protocol import CommunicationProtocol, MultiConditionSafetyEvaluator

class MissionControllerServer:
    """Socket server simulating Mission Controller receiving telemetry from AI companion computer."""

    def __init__(self, host="127.0.0.1", port=8888, conf_threshold=0.70, hardware_trigger_enabled=False):
        self.host = host
        self.port = port
        self.conf_threshold = conf_threshold
        self.hardware_trigger_enabled = hardware_trigger_enabled

        self.server_socket = None
        self.is_running = False
        self.protocol = CommunicationProtocol()
        self.safety_evaluator = MultiConditionSafetyEvaluator(conf_threshold, hardware_trigger_enabled)

        self.received_messages_count = 0
        self.approved_sprays_count = 0
        self.denied_sprays_count = 0

    def start(self):
        """Starts socket server listener thread."""
        self.is_running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"[MissionControllerServer] Socket server listening on {self.host}:{self.port} (Hardware Trigger Enabled={self.hardware_trigger_enabled}).")

    def _listen_loop(self):
        """Main connection listening loop."""
        while self.is_running:
            try:
                self.server_socket.settimeout(1.0)
                client_sock, addr = self.server_socket.accept()
                print(f"[MissionControllerServer] Client connected from {addr}.")
                self._handle_client(client_sock)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[MissionControllerServer] Listener error: {e}")
                break

    def _handle_client(self, client_sock):
        """Handles incoming socket stream from AI companion computer."""
        client_sock.settimeout(2.0)
        buffer = ""
        while self.is_running:
            try:
                data = client_sock.recv(4096)
                if not data:
                    print("[MissionControllerServer] Client disconnected.")
                    break
                buffer += data.decode("utf-8")
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                        
                    msg_dict, status = self.protocol.decode_message(line.encode("utf-8"))
                    if msg_dict is None:
                        print(f"[MissionControllerServer] Message decode/dedup rejected: {status}")
                        continue

                    self.received_messages_count += 1

                    # Evaluate Multi-Condition Safety Gate
                    comm_ok = self.protocol.is_communication_healthy()
                    approved, s_status, action, conds = self.safety_evaluator.evaluate_spray_request(
                        msg_dict, drone_state_valid=True, spray_armed=True, comm_healthy=comm_ok
                    )

                    if approved:
                        self.approved_sprays_count += 1
                    else:
                        self.denied_sprays_count += 1

                    print(f"[MissionControllerServer] Seq {msg_dict['sequence_id']}: {s_status} -> {action}")

                    # Send ACK response
                    ack_payload = {
                        "ack_sequence_id": msg_dict["sequence_id"],
                        "status": s_status,
                        "action": action,
                        "timestamp": time.time()
                    }
                    client_sock.sendall((json.dumps(ack_payload) + "\n").encode("utf-8"))

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[MissionControllerServer] Client handler exception: {e}")
                break

        client_sock.close()

    def stop(self):
        """Stops socket server and releases resources."""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        print("[MissionControllerServer] Server stopped.")

if __name__ == "__main__":
    server = MissionControllerServer(port=8888)
    server.start()
    time.sleep(2.0)
    server.stop()

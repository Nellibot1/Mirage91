# controller.py

import config


KEYBOARD_COMMAND_MAP = {
    config.BLINK_COMMAND: "down",
    config.CLENCH_COMMAND: "space",
}


class Controller:

    def __init__(self, mode="print"):
        self.mode = mode
        self._setup()

    def _setup(self):
        if self.mode == "keyboard":
            from pynput.keyboard import Controller as KeyController, Key
            self.keyboard = KeyController()
            self.Key = Key
        elif self.mode == "socket":
            import socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.game_address = (config.COMMAND_HOST, config.COMMAND_PORT)
            print(f"Socket controller ready -> {self.game_address}")
        elif self.mode == "serial":
            import serial
            self.serial_conn = serial.Serial(port="COM3", baudrate=9600)

    def send_prediction(self, predicted_class, label):
        print(f"[{label}]", end="  ", flush=True)
        if self.mode == "print":
            print("-> waiting for event")
        else:
            print("", end="")

    def send_event(self, command, source, predicted_class, label):
        print(f"[{label}] -> {command} ({source})")

        if self.mode == "print":
            return
        if self.mode == "keyboard":
            self._send_keyboard(command)
            return
        if self.mode == "socket":
            self._send_socket(command)
            return
        if self.mode == "serial":
            self._send_serial(predicted_class)

    def _send_keyboard(self, command):
        key_name = KEYBOARD_COMMAND_MAP.get(command)
        if key_name is None:
            return

        if key_name == "space":
            key = self.Key.space
        elif key_name == "down":
            key = self.Key.down
        else:
            key = key_name

        self.keyboard.press(key)
        self.keyboard.release(key)

    def _send_socket(self, command):
        self.sock.sendto(command.encode("utf-8"), self.game_address)

    def _send_serial(self, predicted_class):
        self.serial_conn.write(str(predicted_class).encode())

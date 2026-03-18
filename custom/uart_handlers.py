from abc import ABC, abstractmethod
from enum import Enum

# --- 1. Hardware Abstraction Layer ---

class UARTInterface(ABC):
    @abstractmethod
    def send(self, data: bytes) -> None:
        pass

class DummyUART(UARTInterface):
    def send(self, data: bytes) -> None:
        print(f"[UART] Sending data: {data.hex()}")

class PIUART(UARTInterface):
    def __init__(self, port: str = '/dev/serial0', baudrate: int = 9600):
        try:
            import serial
        except ImportError:
            raise ImportError("Please install pyserial to use PIUART (pip install pyserial)")
        self.ser = serial.Serial(port, baudrate)

    def send(self, data: bytes) -> None:
        self.ser.write(data)
    def close(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()



class ConvoyerBeltActions(Enum):
    START = 1
    STOP = 2 

class ConveyorController:
    def __init__(self, uart: UARTInterface):
        self.uart = uart
    def execute_action(self, action: ConvoyerBeltActions) -> None:
        payload = action.value.to_bytes(1, byteorder='big')
        self.uart.send(payload)
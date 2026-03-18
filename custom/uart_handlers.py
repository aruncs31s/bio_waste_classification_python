
from abc import ABC, abstractmethod
class UARTInterface(ABC):
    @abstractmethod
    def send(self, data: bytes) -> None:
        pass

class DummyUART(UARTInterface):
    def send(self, data: bytes) -> None:
        print(f"[UART] Sending data: {data.hex()}")


from enum import Enum
class ConvoyerBeltActions(Enum):
    START = 1
    STOP = 2 
   
class PIUART(UARTInterface):
    
    def __init__(self, port: str = '/dev/serial0', baudrate: int = 9600):
        try:
            import serial
            self.ser = serial.Serial(port, baudrate)
            
        except Exception as e:
            raise ImportError("Please install pyserial to use PIUART")
    def send(self, data: ConvoyerBeltActions) -> None:
        self.ser.write(data.value.to_bytes(1, byteorder='big'))
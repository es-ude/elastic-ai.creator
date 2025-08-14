from enum import IntEnum


class Command(IntEnum):
    NAK = 0
    ACK = 1
    READ_SKELETON_ID = 2
    GET_FLASH_CHUNK_SIZE = 3
    WRITE_TO_FLASH = 4
    READ_FROM_FLASH = 5
    FPGA_POWER = 6
    FPGA_LEDS = 7
    MCU_LEDS = 8
    INFERENCE = 9
    DEPLOY_MODEL = 10

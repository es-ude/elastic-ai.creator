# FPGA IO

## Memmory Mappd IO

**Control Region: 0x00 - 0xFF:**
- LED: 0x03 (1 byte)
  - each of the lowest four bits control one of the LEDs
  - 0=off, 1=on
- USERLOGIC_CONTROL: 0x04 (1 byte)
  - lowest bit enables (1) / disables (0) the skeleton
- Multiboot: 0x05 - 0x07 (3 bytes)
  - start address of the configuration to load from flash
  - triggers reconfiguration after write to 0x07 is complete
  - always write all three bytes
  - starting with the lowest byte of the address to 0x05
- other addresses are reserved for future uses

**User Logic Region: 0x100 - ??**
- 
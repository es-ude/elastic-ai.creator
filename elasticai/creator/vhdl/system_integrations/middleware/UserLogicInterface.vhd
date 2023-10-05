--
--	Package File Template
--
--	Purpose: This package defines supplemental types, subtypes, 
--		 constants, and functions 
--
--   To use any of the example code shown below, uncomment the lines and modify as necessary
--

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use ieee.numeric_std.all;

package UserLogicInterface is
    -- QXI interface
    constant XMEM_OFFSET			: unsigned(15 downto 0) := x"0000";
    constant XMEM_USERLOGIC_OFFSET  : unsigned(15 downto 0) := x"0100";
    
	
	subtype uint32_t	is unsigned(31 downto 0);
	subtype uint24_t 	is unsigned(23 downto 0);
	subtype uint16_t 	is unsigned(15 downto 0);
	subtype uint8_t	is unsigned(7 downto 0);
	
	subtype int32_t	is signed(31 downto 0);
	subtype int24_t 	is signed(23 downto 0);
	subtype int16_t 	is signed(15 downto 0);
	subtype int8_t		is signed(7 downto 0);

	constant top : integer := 255;
	
	type uint32_t_interface is record
		data 	: uint32_t;
		ready : std_logic;
		-- done	: std_logic;
	end record;
	
	type uint24_t_interface is record
		data 	: uint24_t;
		ready : std_logic;
		-- done	: std_logic;
	end record;
	
	type uint8_t_interface is record
		data 	: uint8_t;
		ready : std_logic;
		-- done	: std_logic;
	end record;
	
	function little_endian (input : natural) return uint32_t;

end UserLogicInterface;

package body UserLogicInterface is

	function little_endian (input : natural) return uint32_t is
	variable big_endian : uint32_t;
	begin
		big_endian := to_unsigned(input, 32);
		return big_endian(7 downto 0) & big_endian(15 downto 8) & big_endian(23 downto 16) & big_endian(31 downto 24);	
	end;

end UserLogicInterface;